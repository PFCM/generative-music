"""
Midi Controlled Sampler, with pluggable ways of changing the samples.
Has sound files associated with a subset of MIDI notes and will listen
to a MIDI bus to decide when they should be started and stopped.
Separately will run a background thread that may occasionally decide to replace
some of the samples with other samples.
"""
import os
import sys
import time

import networkx as nx
import pyo

from sample_walker import (embedding_neighbours_graph, random_walkers,
                           read_embeddings)


def _is_noteon(status):
    """Check if a midi status is a note on event"""
    return (status >> 4) == 9


def _is_noteoff(status, data2):
    """Check if a midi status is a note off event, or a note on with 0 velocity
    as this is sometimes used instead of a noteoff."""
    return ((status >> 4) == 8) or (((status >> 4) == 9) and (data2 == 0))


def make_callback(players, mix):
    """Unclear if this will mirror changes in `players` yet"""

    def _callback(status, data1, data2):
        """Handle a midi event"""
        # print('midi: {}-{}-{}'.format(status, data1, data2), end='')
        if data1 in players:
            if _is_noteoff(status, data2):
                players[data1].stop()
            elif _is_noteon(status):
                vel = pyo.rescale(data2, 0, 127)
                players[data1].setMul(vel)
                players[data1].play()
                mix.out()
        # print()

    return _callback


def listen(device_num, initial_files):
    """Start listening for MIDI. This function will not block, just
    start the audio serber and all the midi handling."""
    server = pyo.Server(nchnls=1)
    server.deactivateMidi()
    server.boot().start()
    players = {
        note: pyo.SfPlayer(path, mul=.1).stop()
        for note, path in initial_files.items()
    }
    mixdown = pyo.Mix(list(players.values()))
    out = pyo.Compress(mixdown)
    out = pyo.Freeverb(out).out()

    raw = pyo.MidiListener(make_callback(players, out), device_num).start()

    return players, server


def main():
    """check that we can listen to midi and play some drums"""
    sample_path = os.path.dirname(sys.argv[1])

    instruments = {
        36: 'Casio MT-100/00.wav',
        42: 'Casio MT-100/01.wav',
        46: 'Casio MT-100/02.wav',
        38: 'Casio MT-100/03.wav',
        47: 'Casio MT-500/09.wav',
        43: 'Casio MT-500/10.wav',
        51: 'Casio MT-500/11.wav',
        57: 'Casio Rapman/06.wav'
    }
    embeddings = read_embeddings(sys.argv[1])

    graph = embedding_neighbours_graph(*embeddings)
    print('{} components'.format(nx.number_connected_components(graph)))

    players, server = listen(
        0, {k: os.path.join(sample_path, v)
            for k, v in instruments.items()})

    def change_player(num, newpath):
        """switch one out"""
        oldpath = players[num].path
        players[num].path = os.path.join(sample_path, newpath)
        print('change {}: {}->{}'.format(num, oldpath, newpath))

    times = {k: 10 for k in instruments}

    random_walkers(graph, instruments, times, change_player)


if __name__ == '__main__':
    main()
