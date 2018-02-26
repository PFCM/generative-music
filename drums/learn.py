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


def main():
    """Load up some pre-transformed drum pattern vectors and see if we can
    have a little bit of a dictionary learn on them"""
    data = np.load('../drums/drum-midi.npy')
    data = data.astype(np.float)
    if not os.path.exists('dictionary_learner.pkl'):
        decomp = sklearn.decomposition.NMF(
            n_components=16,
            l1_ratio=0.5,
            alpha=0.5,
            solver='cd',
            verbose=True,
        )
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
        end = time.time()
        joblib.dump(nn_graph, 'nn_graph.pkl')
        print('got graph, {}s'.format(end - start))
    else:
        print('found existing graph')
        nn_graph = joblib.load('nn_graph.pkl')

    # transform back
    origs = decomp.inverse_transform(coeffs)
    print('MSE: {}'.format(np.mean((origs - data)**2)))


if __name__ == '__main__':
    main()
