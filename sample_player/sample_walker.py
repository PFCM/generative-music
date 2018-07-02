"""
Load some embeddings and do a random walk through them.
"""
import itertools
import os
import sys
import time
from functools import partial
from itertools import chain, islice
from multiprocessing.dummy import Pool

import librosa
import networkx as nx
import numpy as np
import pyo
from sklearn.neighbors import kneighbors_graph

from organise import read_embeddings


def embedding_neighbours_graph(filenames, embeddings, neighbours=2):
    """Make a networkx graph (with filenames as nodes) with connectivity
    derived from the nearest neighbours of `embeddings`.
    Uses euclidean distance."""
    graph = nx.Graph(
        kneighbors_graph(embeddings, neighbours, p=2, mode='distance'))
    graph = nx.relabel_nodes(
        graph, {i: f
                for i, f in enumerate(filenames)}, copy=False)
    return graph


def printer(gen, ident=''):
    """print the elements of a generator then yield them"""
    for item in gen:
        print('{}{}'.format(ident, item))
        yield item


def random_walk(graph, start_node):
    """Random walk through the graph, yielding nodes as we go. Will be
    infinite, use itertools.islice or the like to get the right amount."""
    current_node = start_node
    while True:
        yield current_node

        options = list(graph.neighbors(current_node))
        current_node = np.random.choice(options)


def random_walkers(graph, start_nodes, times, callback):
    """Run a bunch of random walkers with exponentially distributed step
    times. Blocks until a keyboard interrupt.

    `start_nodes` and `times` should be dictionaries, the keys will be passed
    to `callback` along with the new values.
    """
    keys, starters = zip(*start_nodes.items())
    rates = np.array([times[k] for k in keys])
    wait_times = np.random.exponential(rates)
    walkers = [random_walk(graph, start_nodes[key]) for key in keys]
    while True:
        try:
            wait = np.min(wait_times)
            time.sleep(wait)
            wait_times -= wait
            changes, = np.where(wait_times < 1e-7)
            for result in changes:
                callback(keys[result], next(walkers[result]))
                wait_times[result] = np.random.exponential(rates[result])
        except KeyboardInterrupt:
            return


def walker(graph, args, length=10):
    """Random walk through a component of the sound graph, playing as we go"""
    num, component = args
    start = np.random.choice(list(component))
    files = islice(random_walk(graph, start), max(5, len(component) // length))
    print('{}--{}'.format(num, len(component)))
    files = printer(files, '~~~~{}~'.format(num))
    samples = (os.path.join(os.path.dirname(sys.argv[1]), path)
               for path in files)
    player = pyo.SfPlayer(next(samples), mul=0.1)

    trig = pyo.TrigFunc(player['trig'], make_looper(samples, player))
    player.out()
    while player.isPlaying():
        time.sleep(1)


def make_looper(gen, player):
    """make a trigger function to loop through the generator"""

    def _replace_fname():
        player.stop()
        try:
            player.setPath(next(gen))
            player.out()
        except StopIteration:
            pass

    return _replace_fname


def main():
    """quick test"""
    embeddings = read_embeddings(sys.argv[1])

    graph = embedding_neighbours_graph(*embeddings)
    print('{} components'.format(nx.number_connected_components(graph)))

    server = pyo.Server(nchnls=1, duplex=0).boot()
    server.start()
    with Pool(8) as pool:
        results = pool.imap_unordered(
            partial(walker, graph), enumerate(nx.connected_components(graph)))
        _ = list(results)


if __name__ == '__main__':
    main()
