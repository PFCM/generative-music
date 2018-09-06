"""
Actually load the nice clean encoded files into a tensorflow Dataset for
easy consumption.
"""
import json

import numpy as np
import tensorflow as tf

from drums.data import Note


def _encode_track(track):
    """Turn a list of ints into a dictionary of arrays of ints for the
    various parts."""
    notes = [Note.from_int(n) for n in track]
    return {
        'note': np.array([n.note_num for n in notes]),
        'vel': np.array([n.velocity for n in notes]),
        'len': np.array([n.length for n in notes]),
        'delta': np.array([n.delta for n in notes])
    }


def make_dataset(path, max_length, batch_size, front_pad=0, one_hot=True):
    """Load the json encoded results of convert_data.py into a tensorflow
    dataset yielding dictionaries of the various components. The dictionary
    will have keys corresponding parallel time series for the following:
        - "note": the note of the event
        - "vel": the velocity of the event
        - "len": the length of the event
        - "delta": the time elapsed since the previous event

    Args:
        path: path to the json containing the encoded data.
        max_length: maximum length of items in the dataset
        batch_size: size of the batches of data desired
        front_pad: extra padding to add to the front of the batches, to account
            for the receptive field of the network.
        one_hot: whether or not to make the data one-hots of appropriate size
            or to just leave it as integers in the appropriate range. The
            dimensions of the one-hots are currently fixed to the drum
            representation described in `drums/data.py`

    Returns:
        Dataset: generating dictionaries as described.
    """
    # first load all the data and make sure it's sane
    with open(path) as fhandle:
        tracks = json.load(fhandle)['tracks']
    raw_data = [t['data'] for t in tracks]
    raw_data = [t for t in raw_data if t is not None and len(t)]
    # for now just keep the encoded tracks in memory
    # if this is looking like a lot of memory then we'll sort that out later
    encoded_tracks = [_encode_track(t) for t in raw_data]

    dataset = tf.data.Dataset.from_generator(lambda: encoded_tracks, {
        'note': tf.int64,
        'vel': tf.int64,
        'len': tf.int64,
        'delta': tf.int64
    }, {
        'note': [None],
        'vel': [None],
        'len': [None],
        'delta': [None]
    })

    if one_hot:
        dataset = dataset.map(_make_onehot)

    sizes = {'note': 8, 'vel': 32, 'len': 4, 'delta': 64}
    padded_shapes = {k: [max_length, s] for k, s in sizes.items()}

    dataset = dataset.padded_batch(batch_size, padded_shapes)
    # and now pad again for the receptive field
    if front_pad > 0:

        def _pad_front(items):
            """pad the appropriate amount at the beginning of the time axis"""
            return {
                k: tf.pad(v, [[0, 0], [front_pad, 0], [0, 0]])
                for k, v in items.items()
            }

        dataset = dataset.map(_pad_front)
    return dataset


def _make_onehot(items):
    """Turn the various components into onehots."""
    sizes = {'note': 8, 'vel': 32, 'len': 4, 'delta': 64}

    return {k: tf.one_hot(v, sizes[k]) for k, v in items.items()}
