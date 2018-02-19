"""Dealing with the data. AKA load some midi files and write
something a bit more normal then read said normal."""
import itertools
import logging
from collections import defaultdict

import mido
import numpy as np
from six.moves import reduce

# The instruments we care about
KICK = 0
SNARE = 1
HIHAT_OPEN = 2
HIHAT_CLOSED = 3
CRASH = 4
RIDE = 5
TOM_HIGH = 6
TOM_LOW = 7

# INSTRUMENT MAPPING
INSTRUMENT_MAP = defaultdict(
    lambda: None,
    {
        35: KICK,  # bass drum 1
        36: KICK,  # bass drum 2
        37: None,  # rimshot
        38: SNARE,  # snare drum 1
        39: None,  # hand clap
        40: SNARE,  # snare drum 2
        41: TOM_LOW,  # low tom 2
        42: HIHAT_CLOSED,  # closed hihat
        43: TOM_LOW,  # low tom 1
        44: HIHAT_CLOSED,  # pedal hi-hat
        45: TOM_LOW,  # mid tom 2
        46: HIHAT_OPEN,  # open hi-hat
        47: TOM_HIGH,  # mid tom 1
        48: TOM_HIGH,  # high tom 2
        49: CRASH,  # crash 1
        50: TOM_HIGH,  # tom 2
        51: RIDE,  # ride 1
        52: CRASH,  # china
        53: RIDE,  # ride bell
        54: HIHAT_OPEN,  # tambourine
        55: CRASH,  # splash
        56: None,  # cowbell
        57: CRASH,  # crash 2
        58: None,  # "vibra slap"
        59: RIDE,  # ride 2
        60: TOM_HIGH,  # high bongo
        61: TOM_LOW,  # low bongo
        62: TOM_HIGH,  # mute high conga
        63: TOM_HIGH,  # open high conga
        64: TOM_LOW,  # low conga
        65: TOM_HIGH,  # high timbale
        66: TOM_LOW,  # low timbale
        67: None,  # high agogo
        68: None,  # low agogo
        69: None,  # cabasa
        70: None,  # maracas
        71: None,  # short whistle
        72: None,  # long whistle
        73: None,  # short guiro
        74: None,  # long guiro
        75: None,  # claves
        76: None,  # high wood block
        77: None,  # low wood block
        78: None,  # mute cuica
        79: None,  # open cuica
        80: None,  # mute triangle
        81: None,  # open triangle
    })

IGNORED_MESSAGES = [  # midi stuff we don't care about
    'copyright',  # lol?
    'track_name',
    'sequence_number',
    'text',
    'instrument_name',
    'lyrics',
    'marker',
    'cue_marker',
    'device_name',
    'channel_prefix',
    'midi_port',
    'set_tempo',  # maybe one day,
    'smpte_offset',
    'key_signature',  # this is just drums atm
    'sequencer_specific',  # we are non-specific
    'unknown_meta'
]


class TimeSignatureException(Exception):
    pass


def _ignore(data, msg):
    """ignore a msg, but log it"""
    logging.info('ignoring message: %s', msg)
    return data


def _dict_update(data, **kwargs):
    """less OO dict.update()"""
    data.update(**kwargs)
    return data


def _note_on_update(data, msg):
    """Update the accumulated data with a new note"""
    if msg.channel == 9:  # GM drum channel only
        data = _write_for(data, msg.time)
        data['active_notes'][INSTRUMENT_MAP[msg.note]] = msg.velocity
    else:
        data = _ignore(data, msg)
    return data


def _note_off_update(data, _):
    """for now just ignore them"""
    return data


def _write_for(data, ticks):
    """Write notes into the sequence for a given number of ticks"""
    # quantise the time to sixteenth notes
    ticks_per_sixteenth = data['ticks_per_beat'] / 16
    time_delta = int(round(ticks / ticks_per_sixteenth))
    if time_delta != ticks:
        logging.info('quantised %d ticks to %d', ticks,
                     time_delta * ticks_per_sixteenth)
    if time_delta > 0 and 'active_notes' in data:
        active_notes = data['active_notes']
        if 'sequence' not in data:
            data['sequence'] = []
        for _ in range(0, time_delta):
            data['sequence'].append(active_notes)
            active_notes = np.zeros(8, dtype=np.int)
        data['active_notes'] = active_notes
    elif 'active_notes' not in data:
        data['active_notes'] = np.zeros(8, dtype=np.int)
    return data


def _flush(data, msg):
    """flush any active notes, finalising the data"""
    data = _write_for(data, msg.time)
    return data


def _read_message(data, msg):
    """Read a single midi message into an accumulating data
    dictionary. Will freak out if the meta messages don't show up
    before notes."""
    if msg.type in IGNORED_MESSAGES:
        data = _ignore(data, msg)
    elif msg.type == 'time_signature':
        # NOTE: right now we're only handling fours
        if msg.numerate == 4 and msg.denominator == 4:
            data = _dict_update(
                data,
                clocks_per_click=msg.clocks_per_click,
                notated_32nd_notes_per_beat=msg.notated_32nd_notes_per_beat)
        else:
            raise TimeSignatureException('not 4/4')
    elif msg.type == 'note_on':
        data = _note_on_update(data, msg)
    elif msg.type == 'note_off':
        data = _note_off_update(data, msg)
    elif msg.type == 'end_of_track':
        data = _flush(data, msg)

    return data


def read_file(path):
    """Read midi file at `path` and return a generator of numpy
    arrays for each bar.

    Currently only works with 4/4 drum patterns and only extracts
    some of the key general midi drum instruments on channel 10.

    We also expect a meta message with time signature info to show
    up before any notes, otherwise it's a bit hard to sort it out.

    Args:
        path (str): path to the file to read.

    Yields:
        numpy vectors, zero or more depending the file.
    """
    mid = mido.MidiFile(path)
    # reading the notes and collecting is a fold
    try:
        results = reduce(_read_message, mid.tracks[1],
                         {'ticks_per_beat': mid.ticks_per_beat})
        # if the sequence is the wrong size, either repeat or chop
        print('\n'.join([
            ','.join(['{:>3}'.format(item) for item in vec])
            for vec in results['sequence']
        ]))
        print(len(results['sequence']))

        if len(results['sequence']) != 256:
            logging.info('sequence is the wrong length!')
        return [np.concatenate(results['sequence'])]
    except TimeSignatureException as exc:
        logging.info('%s: wrong time signature', path)
        return []


def main():
    import sys
    logging.getLogger().setLevel(logging.DEBUG)
    results = list(itertools.chain.from_iterable(map(read_file, sys.argv[1:])))
    print(len(results))
    lens = np.array([v.shape[0] for v in results])
    print(np.mean(lens), np.min(lens), np.max(lens))


if __name__ == '__main__':
    main()
