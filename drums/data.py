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
    'end_of_track',  # figure this out ahead of time
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


def ticks_to_sixteenth(ticks, ticks_per_beat, nearest=True):
    """Given the resolution info from a midi file (ticks per beat) convert
    a time in ticks to a time in sixteenth notes, optionally the nearest
    integral one."""
    ticks_per_sixteenth = ticks_per_beat / 4
    time_delta = ticks / ticks_per_sixteenth
    if nearest:
        time_delta = int(round(time_delta))
    return time_delta


def _note_on_update(data, msg):
    """Update the accumulated data with a new note"""
    if msg.channel == 9:  # GM drum channel only
        # quantise the time to sixteenth notes
        ftime_delta = ticks_to_sixteenth(msg.time, data['ticks_per_beat'],
                                         False)
        index = int(round(data['current_time'] + ftime_delta))
        data['sequence'][index, INSTRUMENT_MAP[msg.note]] = msg.velocity
        data['current_time'] += ftime_delta
    else:
        data = _ignore(data, msg)
    return data


def _note_off_update(data, msg):
    """for now just ignore them"""
    data['current_time'] += ticks_to_sixteenth(msg.time,
                                               data['ticks_per_beat'], False)
    return data


def _read_message(data, msg):
    """Read a single midi message into an accumulating data
    dictionary. Will freak out if the meta messages don't show up
    before notes."""
    if msg.type in IGNORED_MESSAGES:
        data = _ignore(data, msg)
    elif msg.type == 'time_signature':
        # NOTE: right now we're only handling fours
        if msg.numerator == 4 and msg.denominator == 4:
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

    return data


def chunk_iterator(data, chunk_size):
    """break up a sequence longer than `chunk_size` into segments
    of `chunk_size`, discarding any leftover"""
    for i in range(0, len(data), chunk_size):
        if len(data) - i >= chunk_size:
            yield data[i:i + chunk_size]


def repeat_or_chunk(data, chunk_size):
    """Either repeat some data until there are chunk_size elements
    otherwise split into chunk_size pieces. Or nothing, rarely"""
    if len(data) < chunk_size:
        repeats = chunk_size // len(data)
        if repeats != chunk_size / len(data):
            logging.info('skipping something that does not divide four bars')
            data = []
        else:
            data = data * repeats
        yield data
    else:
        return chunk_iterator(data, chunk_size)


def _get_tempo(msgs):
    """Search through msgs for a meta message with type 'set_tempo' and return
    the tempo of the last one found."""
    msgs = list(filter(lambda m: m.type == 'set_tempo', msgs))
    if msgs:
        return mido.tempo2bpm(msgs[-1].tempo)
    return 120  # by the standard, default to 120


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
        beats_per_minute = _get_tempo(mid.tracks[0] + mid.tracks[1])
        len_in_beats = int(round((mid.length / 60) * beats_per_minute))
        logging.info('track is %d beats long', len_in_beats)
        seq = np.zeros((len_in_beats * 4, 8), dtype=np.int)
        results = reduce(
            _read_message, mid.tracks[1], {
                'ticks_per_beat': mid.ticks_per_beat,
                'sequence': seq,
                'current_time': 0
            })
        # if the sequence is the wrong size, either repeat or chop
        print('\n'.join([
            ','.join(['{:>3}'.format(item) for item in vec])
            for vec in results['sequence']
        ]))
        print(len(results['sequence']))

        results = filter(lambda seq: seq != [],
                         repeat_or_chunk(results['sequence'], 256))
        return map(lambda x: np.concatenate(list(x)), results)

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
