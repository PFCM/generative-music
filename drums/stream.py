"""ride the wave"""
import argparse
import sys
from functools import partial as p

import mido
import numpy as np
from sklearn.externals import joblib

import data as data
import learn as learn


def _uncompress(coeffs, model):
    """
    Use a reloaded sklearn model to turn a squished set of coefficients into
    a big explicit 4 bar vector.
    Wrapped up because not all the sklearn models have an inverse_transform
    function.
    """
    if hasattr(model, 'inverse_transform'):
        return learn.sanitise_midi_vector(model.inverse_transform(coeffs))
    if hasattr(model, 'components_'):
        return learn.sanitise_midi_vector(np.dot(coeffs, model.components_))
    raise TypeError('Do not know how to invert {}'.format(model))


def _coeffs_to_mfile(coeffs, model, tempo):
    """
    Turn some coefficients into a MidiFile mido will play for us
    """
    return data.make_midi_file(
        data.vec_to_notes(_uncompress(coeffs, model)), tempo=tempo)


def _play(coeffs, model, tempo, port):
    """
    Blow up some squished coefficients, turn it to midi and spit it out.
    Blocks until it's played.
    """
    play_midifile(port, _coeffs_to_mfile(coeffs, model, tempo), 16)


def parse_args(args=None):
    """Get command line args"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--nn_graph', help='path to the saved nearest neighbour graph')
    parser.add_argument(
        '--decomposition', help='path to the saved sklearn decomposition')
    parser.add_argument('--port', help='name of the midi port')
    parser.add_argument('--tempo', type=float, help='tempo in bpm')

    return parser.parse_args(args)


def play_midifile(port, mfile, max_time=None):
    """
    Play a mido.MidiFile through a port.
    Takes as long as the file takes and returns immediately.
    Unless you've specified the beats argument, in which case we will stop
    early or wait until that many beats have been played.
    """
    total_time = 0

    for msg in mfile.play():
        total_time += msg.time
        port.send(msg)
        if max_time and total_time >= max_time:
            break


def _iterate(func, init):
    """bit like in haskell prelude, but also stops if func returns None"""
    val = init
    while val is not None:
        yield val[1]
        val = func(val)


def play_trajectory(start, step_fn, step_data, model, tempo, port):
    """
    Play a trajectory of some sort. Lasts until it is done or a keyboard
    interrupt.

    Args:
        start: initial coefficients.
        step_fn: callable taking extra data and a vector of coefficients
            that returns either a news set of coefficients to be played or None
            to halt playback.
        step_data: something passed to step_fn as the first argument.
        model: skelarn decomposition used to turn the coefficients into midi.
        port: mido output port to play midi through.
    """
    for coeffs in _iterate(p(step_fn, step_data), start):
        try:
            _play(coeffs, model, tempo, port)
        except KeyboardInterrupt:
            print('\n\nKeyboard interrupt, stopping playback.')
            return


def _random_graph_step(graph, state):
    """take one whole graph step to a random neighbour each time"""
    node, _ = state
    next_node = np.random.choice(graph[node])
    print('{} -> {}'.format(node, next_node))
    return next_node, graph.node[next_node]['coefficients']


def _partial_step(a, b, rate, pos):
    """Just linear interpolation, returns args for next time"""
    new_pos = min(pos + rate, 1.0)
    new_val = a + new_pos * (b - a)
    return new_val, new_pos, new_pos == 1


def main():
    """
    Grab a midi port, a tempo and some kind of patterny info and get cracking
    """
    args = parse_args()
    nn_graph = joblib.load(args.nn_graph)
    decomp = joblib.load(args.decomposition)
    init = np.random.choice(nn_graph)
    init = (init, nn_graph.node[init]['coefficients'])
    with mido.open_output(args.port) as port:
        play_trajectory(init, _random_graph_step, nn_graph, decomp, args.tempo,
                        port)


if __name__ == '__main__':
    main()
