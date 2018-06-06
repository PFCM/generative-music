"""
Helpers to try and clean the data even more by breaking it into chunks if it
is too long.
"""
import functools
from itertools import chain

import mido


def _num_note_ons(mfile):
    """Counts the number of note on events in a midifile (summed across all
    tracks)"""
    return sum(
        len(list(filter(lambda m: m.type == 'note_on', track)))
        for track in mfile.tracks)


def _num_bars(mfile):
    """count the number of bars in a midi file. Not trivial because the time
    signature is a meta-message that might not be there or might change.
    """
    return max(_track_num_bars(t, mfile.ticks_per_beat) for t in mfile.tracks)


def _track_num_bars(track, ticks_per_beat):
    """Count the number of bars in a single track"""
    bars = 0
    beats_per_bar = 4  # default to 4/4
    bars_per_tick = 1 / (beats_per_bar * ticks_per_beat)
    for msg in track:
        bars += msg.time * bars_per_tick
        if msg.type == 'time_signature':
            beats_per_bar = msg.numerator / (msg.denominator / 4.)
            bars_per_tick = 1 / (beats_per_bar * ticks_per_beat)
    return bars


def _track_absolute_time(track):
    """
    Compute the absolute time of all of the events in a track.
    Returns a list of (time, msg) tuples where.
    """

    def _collect(agg, msg):
        """collect the msgs with running times"""
        agg.append((agg[-1][0] + msg.time, msg))
        return agg

    return functools.reduce(_collect, track[1:], [(0, track[0])])


def binary_search(val, items, key=None):
    """
    Find the index of the first item in `items` that is >= than `val`.
    Assumes `items` is sorted.

    Args:
        val: the value we are looking for.
        items: a sequence of values to search.
        key: optional function to transform an item in the sequence into
            something we can compare against `val`.

    Returns:
        the index into which `val` would be inserted to keep the list in
            order.
    """
    bottom, top = 0, len(items)
    while bottom < top:
        idx = int((top + bottom) / 2)
        test_val = items[idx]
        if key is not None:
            test_val = key(test_val)

        if test_val == val:
            return idx
        elif test_val < val:
            bottom = idx + 1
        else:
            top = idx
    return bottom


def _midifile_like(mfile, track=None):
    """Make a midi file with the same attributes, potentially adding a new
    track."""
    newfile = mido.MidiFile(ticks_per_beat=mfile.ticks_per_beat)
    if track is not None:
        newfile.add_track(track.name)
        newfile.tracks[0].extend(track)
    return newfile


def _split_in_half(mfile):
    """Find the halfway point, in musical terms, and split it up, returning
    both tracks. If it doesn't end with the right meta message this could be
    trouble >.>."""
    # TODO powers of 2 or something...
    absolute_times = _track_absolute_time(mfile.tracks[0])
    halfway_time = absolute_times[-1][0] / 2.
    split_idx = binary_search(halfway_time, absolute_times, key=lambda x: x[0])
    track_a = mfile.tracks[0][:split_idx]
    track_a.name = track_a.name + '-0'
    track_b = mfile.tracks[0][split_idx:]
    track_b.name = track_b.name + '-1'
    return (_midifile_like(mfile, mfile.tracks[0][:split_idx]),
            _midifile_like(mfile, mfile.tracks[0][split_idx:]))


def split_file(mfile, max_bars=32, max_notes=256):
    """Splits a file into non-overlapping chunks with less than `max_notes`
    'note on' events. Each chunk will be a power of two bars, at most
    `max_bars`. Proceeds recursively until we have chunks with both conditions
    met and then attempts to remove duplicates.

    Args:
        mfile: the midi file to split, a mido.MidiFile with a single track.
        max_bars (int): the maximum number of bars we are allowed.
        max_notes (int): the maximum number of notes we are allowed.

    Yields:
        mido.MidiFile: midi files with all the same meta information as `mfile`
            but no more than `max_bars` worth of time covered and no more than
            `max_notes` note on events in their single track.

    Raises:
        ValueError: if the midi file has more than one track.
    """
    if len(mfile.tracks) != 1:
        raise ValueError('can only handle one midi track, got {}'.format(
            len(mfile.tracks)))
    print('bars: {}, notes: {}'.format(_num_bars(mfile), _num_note_ons(mfile)))
    # do we have to split at all?
    if _num_bars(mfile) > max_bars or _num_note_ons(mfile) > max_notes:
        first, last = _split_in_half(mfile)
        return chain(
            split_file(first, max_bars, max_notes),
            split_file(last, max_bars, max_notes))
    return (mfile, )
