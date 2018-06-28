"""
Load some embeddings and do a random walk through them.
"""
import itertools
import os
import sys
import time
from functools import lru_cache
from itertools import chain, islice

import librosa
import networkx as nx
import numpy as np
import pyo
from sklearn.neighbors import kneighbors_graph

from organise import read_embeddings


def embedding_neighbours_graph(filenames, embeddings, neighbours=8):
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
        # print('{} choices'.format(len(options)))
        current_node = np.random.choice(options)


def make_looper(gen, player):
    """make a trigger function to loop through the generator"""

    def _replace_fname():
        player.stop()
        print('XO')
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
    server = pyo.Server().boot()
    server.start()
    player = None
    for i, component in enumerate(nx.connected_components(graph)):
        print('~~component {}'.format(i + 1))
        print('~~          {} nodes'.format(len(component)))
        # play some random samples from the component
        start = np.random.choice(list(component))
        files = islice(random_walk(graph, start), 10)
        files = printer(files, '~~~~')
        samples = (os.path.join(os.path.dirname(sys.argv[1]), path)
                   for path in files)
        if player is None:
            player = pyo.SfPlayer(next(samples), loop=False)
        else:
            trig.stop()

        trig = pyo.TrigFunc(player['trig'], make_looper(samples, player))
        player.out()
        while player.isPlaying():
            time.sleep(0.5)


if __name__ == '__main__':
    main()
