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
NOTE_INSTRUMENT_MAP = defaultdict(
    lambda: None,
    {
        35: KICK,  # bass drum 1
        36: KICK,  # bass drum 2
        37: SNARE,  # rimshot
        38: SNARE,  # snare drum 1
        39: SNARE,  # hand clap
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

INSTRUMENT_NOTE_MAP = {
    KICK: 35,
    SNARE: 38,
    HIHAT_OPEN: 46,
    HIHAT_CLOSED: 42,
    CRASH: 57,
    RIDE: 51,
    TOM_HIGH: 47,
    TOM_LOW: 43
}

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
        # it might round off the end
        if index == data['sequence'].shape[0]:
            index -= 1
        if msg.velocity > 0:
            data['sequence'][index, NOTE_INSTRUMENT_MAP[
                msg.note]] = msg.velocity
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
        if (i + chunk_size) <= len(data):
            yield data[i:i + chunk_size]


def repeat_or_chunk(data, chunk_size):
    """Either repeat some data until there are chunk_size elements
    otherwise split into chunk_size pieces. Or nothing, rarely"""
    if len(data) < chunk_size:
        repeats = chunk_size // len(data)
        if (repeats * len(data)) != chunk_size:
            logging.info('skipping something that does not divide four bars')
            data = []
        else:
            data = list(data) * repeats
        return [data]
    return chunk_iterator(data, chunk_size)


def _get_tempo(msgs):
    """Search through msgs for a meta message with type 'set_tempo' and return
    the tempo of the last one found."""
    msgs = list(filter(lambda m: m.type == 'set_tempo', msgs))
    if msgs:
        return mido.tempo2bpm(msgs[-1].tempo)
    return 120  # by the standard, default to 120


def _get_time_signature(msgs):
    """Search for time signature messages, return the value of the first"""
    msg = next(msg for msg in msgs if msg.type == 'time_signature')
    if msg:
        return (msg.numerator, msg.denominator)
    return (4, 4)


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
    beats_per_minute = _get_tempo(mid.tracks[0] + mid.tracks[1])
    time_signature = _get_time_signature(mid.tracks[0] + mid.tracks[1])
    if time_signature == (4, 4):
        len_in_beats = int(np.ceil((mid.length / 60) * beats_per_minute))
        logging.info('track is %d beats long', len_in_beats)
        seq = np.zeros((len_in_beats * 4, 8), dtype=np.int)
        results = reduce(
            _read_message, mid.tracks[1], {
                'ticks_per_beat': mid.ticks_per_beat,
                'sequence': seq,
                'current_time': 0
            })
        # if the sequence is the wrong size, either repeat or chop
        # print('\n'.join([
        #     ','.join(['{:>3}'.format(item) for item in vec])
        #     for vec in results['sequence']
        # ]))

        results = filter(lambda seq: seq != [],
                         repeat_or_chunk(results['sequence'], 64))
        results = list(results)
    else:
        logging.info('skipping non 4/4 track: %s', path)
        results = []
    return map(lambda x: np.concatenate(list(x)), results)


def vec_to_notes(midi_vector, channel=9, ticks_per_beat=48):
    """Turn a vector of drums into note ons and offs"""
    # first lets organise it a bit better
    midi_vector = midi_vector.reshape((-1, 8))
    logging.info('midi vector reshaped to %s', midi_vector.shape)
    # make note ons and offs
    delta = 0
    messages = []
    ticks_per_sixteenth = ticks_per_beat / 4
    # LOL at least make some functions you loser
    for active_notes in midi_vector:
        if np.all(active_notes == 0):
            delta += ticks_per_sixteenth
        else:
            notes, = active_notes.nonzero()
            # note ons
            for note in notes:
                messages.append(
                    mido.Message(
                        'note_on',
                        note=INSTRUMENT_NOTE_MAP[note],
                        velocity=active_notes[note],
                        time=int(delta),
                        channel=channel))
                delta = 0
            # wait a sec
            delta = ticks_per_sixteenth
            # note offs
            for note in notes:
                messages.append(
                    mido.Message(
                        'note_off',
                        note=INSTRUMENT_NOTE_MAP[note],
                        velocity=0,
                        time=int(delta),
                        channel=channel))
                delta = 0
    return messages


def _make_metadata_track(name, time_signature, tempo, length):
    """Make a track of metadata events"""
    return [
        mido.MetaMessage('track_name', name=name or 'unnamed', time=0),
        mido.MetaMessage(
            'time_signature',
            numerator=time_signature[0],
            denominator=time_signature[1],
            time=0),
        mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo), time=0),
        mido.MetaMessage('end_of_track', time=length)
    ]


def make_midi_file(note_events, ticks_per_beat=48, name=None, tempo=120):
    """Make a fully fledged midi file out of a list of note on/off events."""
    mfile = mido.MidiFile()
    mfile.ticks_per_beat = ticks_per_beat
    # to be like most midi files I've looked at, we will make a first track
    # with just metadata (tempo in particular)
    first_track = _make_metadata_track(
        name, (4, 4), tempo, sum(event.time for event in note_events))
    # and the second track is just the notes
    mfile.add_track()
    mfile.add_track()
    mfile.tracks[0].extend(first_track)
    mfile.tracks[1].extend(note_events)
    return mfile


###############################################################################
# If we want to model the data as an irregularly spaced time series (which is
# probably more appropriate given that is literally midi data) then the
# following functions should help.
#
# The core idea is to add time delta information as just another feature.
# This lets us model the sequence auto-regressively but without a fixed
# resolution grid. See https://arxiv.org/pdf/1802.05162.pdf for inspiration,
# although the scheme described here isn't exactly that.
#
# Specifically for drums we accept some quantisation and encode each note
# on/off pair into 4 separate components:
# - note number (0-7 in this case)
# - velocity (0-31)
# - note length (discretised and fairly short, we only allow 16th, 8th, quarter
#   and half notes (ie. 0-3))
# - onset delta, relative t the preceding note (again discretised but with a
#   much wider range of 0-63 sixteenth notes).
# This then allows us to model each note in the data as 3 separate (not
# independent) categorical distributions. Note also that we have
# 8*32*4*64=65536 possible combinations and could therefore encode each note in
# 16 bits if desired. We could then store it in 16 bit integers, or just as 16
# position vectors. If we use a neural Hawkes process
# (https://arxiv.org/pdf/1612.09328.pdf) the model itself never actually sees
# time delta as it is just used to decay the states. We could therefore expand
# out to a one hot over all 8*32*4=1024 possible notes, which wouldn't be
# insane.
#
# Pros:
# - compact
# - handles variable time signatures & sequence lengths
# - extensible to multiple channels, more notes, different amounts of offsets
#   etc.
# Cons:
# - going to get weird with lots of notes starting at the same time because the
#   order of the events is irrelevent if there are multiple with time delta 0.
###############################################################################


def encode_drum_track(mfile):
    """
    Pull out the drums from a midi file and encode them as described above.
    Some midi files have no note offs for drums, so we will just make them
    sixteenth notes.

    For now just returns the encoding of the first track that has notes in it
    as a list of integers.

    Returns None if no tracks had notes.
    """
    for track in mfile.tracks:
        if _has_notes(track):
            return [
                note.to_int()
                for note in pair_notes(track, mfile.ticks_per_beat)
            ]

    return None


def _has_notes(track):
    """check if a track has notes in it"""
    return any(msg.type == 'note_on' for msg in track)


def pair_notes(track, ticks_per_beat=48):
    """
    Take a midi track and collect pairs of note on/off with appropriate time
    accounting.

    Results will be a list of Note objects, appropriately converted/quantised.
    """
    time_offset = 0  # accumulate time since last note_on
    ticks_per_sixteenth = ticks_per_beat // 4
    notes = []
    for i, msg in enumerate(track):
        time_offset += msg.time
        # we only care if it's a note on for a drum we want to keep
        if msg.type == 'note_on' and NOTE_INSTRUMENT_MAP[msg.note]:
            length = _find_note_off(msg.note, track[i:]) or ticks_per_sixteenth
            length = _encode_length(length, ticks_per_sixteenth)
            velocity = _encode_velocity(msg.velocity)
            delta = _encode_delta(time_offset, ticks_per_sixteenth)
            notes.append(
                Note(NOTE_INSTRUMENT_MAP[msg.note], length, velocity, delta))
            time_offset = 0
    return notes


def _encode_delta(delta, ticks_per_sixteenth):
    """
    Turn a time offset into a number for a note
    """
    return int(np.minimum(63, np.round(delta / ticks_per_sixteenth)))


def _decode_delta(delta, ticks_per_sixteenth):
    """
    Turn a time offset as used in the Note class into ticks
    """
    return delta * ticks_per_sixteenth


def _encode_velocity(vel):
    """
    Turn a velocity from a midi number into a more restrained representation
    as described below.
    """
    return int(np.round(np.maximum(0, vel - 3) / 4))


def _decode_velocity(vel):
    """
    Turn a velocity from 0-31 as used in the Note class back into the midi
    range of 0-127
    """
    return vel * 4 + 3


def _encode_length(length_in_ticks, ticks_per_sixteenth):
    """Turn length in file ticks to an encoding as specified below"""
    length_in_sixteenths = np.minimum(
        np.maximum(0, length_in_ticks / ticks_per_sixteenth), 8)
    return int(np.round(np.log2(length_in_sixteenths)))


def _decode_length(length, ticks_per_sixteenth):
    """Decode a length the Note class uses into a length in midi file ticks"""
    length_in_sixteenths = 2**length
    return length_in_sixteenths * ticks_per_sixteenth


def _find_note_off(num, track):
    """
    Finds the first note off with note number `num`. Returns the accumulated
    time offset in ticks or None if we reach the end.
    """
    time_offset = 0
    for msg in track:
        time_offset += msg.time
        if msg.type == 'note_off' and msg.note == num:
            return time_offset
    return None


class Note(object):
    """
    Note wrapper holding the length, note number, velocity and
    time offset with methods to convert to a packed
    representation for writing.
    """

    def __init__(self, note_num, length, velocity, delta):
        """initialise the container. Expected units (if you want to write)
        are:
            - note_num: the drum number, see the constants at the top of this
              file
            - length: the length of the final note is 2**length sixteenth
              notes.
            - velocity: 32 positions which will spread linearly across the midi
              range.
            - delta: sixteenth notes, expected to be between 0-63 inclusive.
        """
        self.note_num = note_num
        self.length = length
        self.velocity = velocity
        self.delta = delta

    def to_int(self):
        """
        Pack the data into the first sixteen bits of an integer.
        The layout is, starting from the least significant bit:
        0: note_num
        1: ..
        2: ..
        3: length
        4: ..
        5: velocity
        6: ..
        7: ..
        8: ..
        9: ..
        a: delta
        b: ..
        c: ..
        d: ..
        e: ..
        f: ..
        """
        num = self.note_num & 0b111
        num |= (self.length & 0b11) << 3
        num |= (self.velocity & 0b11111) << 5
        num |= (self.delta & 0b111111) << 10
        return num

    def __eq__(self, value):
        """it is equal if the values are the same"""
        return (self.note_num == value.note_num and self.length == value.length
                and self.velocity == value.velocity
                and self.delta == value.delta)

    def __repr__(self):
        """handy to get a good message when printed"""
        return 'Note(note_num={}, length={}, velocity={}, delta={})'.format(
            self.note_num, self.length, self.velocity, self.delta)

    @staticmethod
    def from_int(packed):
        """
        Construct a note representation from an integer packed as above
        """
        note = packed & 0b111
        length = (packed >> 3) & 0b11
        velocity = (packed >> 5) & 0b11111
        delta = (packed >> 10) & 0b111111

        return Note(note, length, velocity, delta)


def main():
    import sys
    logging.getLogger().setLevel(logging.WARNING)
    results = list(itertools.chain.from_iterable(map(read_file, sys.argv[1:])))
    data = np.stack(results)
    print('data shape: {}'.format(data.shape))
    np.save('../drums/drum-midi.npy', data)
    reconstituted = make_midi_file(vec_to_notes(data[0]), name='test')
    reconstituted.save('test.mid')


if __name__ == '__main__':
    main()
