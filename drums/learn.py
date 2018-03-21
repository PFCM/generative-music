"""
how do we learn about a drums
"""
import os
import time

import networkx as nx
import numpy as np
import sklearn.decomposition
import sklearn.neighbors
from sklearn.externals import joblib

from data import make_midi_file, vec_to_notes


def sanitise_midi_vector(vec):
    return np.clip(vec.astype(np.int), 0, 127)


def write_comparison(original, reconstructed, basename):
    """Write two files to compare"""
    # sanity check the midi
    original = sanitise_midi_vector(original)
    reconstructed = sanitise_midi_vector(reconstructed)
    orig = make_midi_file(
        vec_to_notes(original), name='{}_original'.format(basename))
    orig.save('{}_original.mid'.format(basename))
    recon = make_midi_file(
        vec_to_notes(reconstructed), name='{}_reconstruction'.format(basename))
    recon.save('{}_reconstruction.mid'.format(basename))


def main():
    """Load up some pre-transformed drum pattern vectors and see if we can
    have a little bit of a dictionary learn on them"""
    data = np.load('../drums/drum-midi.npy')
    data = data.astype(np.float)
    if not os.path.exists('dictionary_learner.pkl'):
        # decomp = sklearn.decomposition.NMF(
        #     n_components=64,
        #     l1_ratio=0.5,
        #     alpha=10.0,
        #     solver='cd',
        #     verbose=True,
        # )
        decomp = sklearn.decomposition.MiniBatchDictionaryLearning(
            n_components=256, alpha=10, n_jobs=1, batch_size=10, verbose=True)
        # decomp = sklearn.decomposition.MiniBatchSparsePCA(
        #     n_components=64,
        #     n_jobs=4,
        #     alpha=10,
        #     ridge_alpha=0.5,
        #     verbose=True,
        #     batch_size=64)
        print('fitting')
        start = time.time()
        coeffs = decomp.fit_transform(data)
        end = time.time()
        joblib.dump(decomp, 'dictionary_learner.pkl')
        print('fitted, {}s, ({})'.format(end - start, coeffs.shape))
    else:
        print('found pre-existing learner')
        decomp = joblib.load('dictionary_learner.pkl')
        coeffs = decomp.transform(data)

    if not os.path.exists('nn_graph.pkl'):
        print('building nearest neighbour graph')
        start = time.time()
        neighbours = sklearn.neighbors.NearestNeighbors(
            n_neighbors=10, n_jobs=-1).fit(coeffs)
        nn_graph = nx.Graph(neighbours.kneighbors_graph(n_neighbors=3))
        # need to relabel the nodes with their coefficients for later
        nx.set_node_attributes(
            nn_graph,
            name='coefficients',
            values={node: coeffs[node]
                    for node in nn_graph})
        end = time.time()
        joblib.dump(nn_graph, 'nn_graph.pkl')
        print('got graph, {}s'.format(end - start))
    else:
        print('found existing graph')
        nn_graph = joblib.load('nn_graph.pkl')

    # transform back
    if hasattr(decomp, 'inverse_transform'):
        origs = decomp.inverse_transform(coeffs)
    else:
        origs = np.dot(coeffs, decomp.components_)
    print('MSE: {}'.format(np.mean((origs - data)**2)))

    print('writing a couple of files for comparison')
    randos = np.random.choice(np.arange(data.shape[0]), 5, replace=False)
    for i, (before, after) in enumerate(zip(data[randos], origs[randos])):
        write_comparison(before, after, i)

    # TODO: also look at the individual components found


if __name__ == '__main__':
    main()
