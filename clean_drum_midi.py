"""
Load some percussion midi and check that it's all on the General Midi drum
channel (10).

Then try and separate out all the repeating patterns and write them as separate
files.
"""
import argparse
import hashlib
import io
import itertools
import multiprocessing
import os
import re
import shutil
from functools import partial

import mido

from drums import break_midi


def _directory_typecheck(path):
    """Check a path points to a directory. If it doesn't exist creates it."""
    if os.path.isfile(path):
        raise argparse.ArgumentTypeError(
            '{} exists and is a file not a directory'.format(path))

    os.makedirs(path, exist_ok=True)  # NOTE: python 3 only
    return path


def parse_args(args=None):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input',
        '-i',
        help='directory of files to clean',
        type=_directory_typecheck)
    parser.add_argument(
        '--recursive',
        '-r',
        action='store_true',
        help='whether or not to recurse into subdirectories')
    parser.add_argument(
        '--output',
        '-o',
        type=_directory_typecheck,
        help='where to store output files')
    return parser.parse_args(args)


def _has_notes(track):
    """Check if a MIDI track has note on or note off messages."""
    return len(
        list(filter(lambda m: m.type in ['note_on', 'note_off'], track))) > 0


def _maybe_process(mfile, num_bars=4):
    """Check to see if the file has a single channel which is channel 10, if
    not make sure it does."""
    # often there will be two tracks, one of which only has metadata and
    # one with notes. For now we are just going to ignore tracks without
    # notes
    mfile.tracks = list(filter(_has_notes, mfile.tracks))
    if len(mfile.tracks) >= 2:
        # not really sure how to handle that :|
        # print(' oh no {} tracks '.format(len(mfile.tracks)))
        # raise SystemExit
        return None
    # We want to make sure that the notes will be all on
    # channel 10 so that it will be treated as drums
    # TODO: make sure the final track still has time signature info etc.
    for track in mfile.tracks:
        if _has_notes(track):
            # force them to channel ten
            for msg in track:
                if hasattr(msg, 'channel'):
                    msg.channel = 10
    return break_midi.split_file(mfile, max_bars=num_bars)


def _hash_midi(mfile):
    """Save the file to bytes and hash it. Return the bytes so we can spit them
    straight to disk."""
    with io.BytesIO() as bio:
        mfile.save(file=bio)
        file_bytes = bio.getvalue()
        hash = hashlib.md5()
        hash.update(file_bytes)
        hash = hash.hexdigest()
        return hash + '.mid', file_bytes


def _hash_and_write(outdir, mfile):
    """Write a midifile with into `outdir` with a name based on its hash."""
    name, filebytes = _hash_midi(mfile)
    newpath = os.path.join(outdir, name)
    if os.path.exists(newpath):
        return '__duplicate__'
    with open(newpath, 'wb') as fhandle:
        fhandle.write(filebytes)
    return name


def process_file(outdir, in_path):
    """Grab a file, maybe tidy it up & split, definitely write it with a new
    name based on its md5 hash"""
    try:
        mfile = mido.MidiFile(in_path)
        mfiles = _maybe_process(mfile)
    except OSError:
        print('could not open {}'.format(in_path))
        mfiles = None
    if mfiles is not None:
        return list(map(partial(_hash_and_write, outdir), mfiles))
    return [in_path]


def generate_paths(dirname, recursive=False):
    """Generate paths, potentially recursively."""
    if recursive:
        gen = itertools.chain.from_iterable(
            map(lambda x: map(partial(os.path.join, x[0]), x[2]),
                os.walk(dirname)))
    else:
        gen = filter(os.path.isfile,
                     map(partial(os.path.join, dirname), os.listdir(dirname)))
    return filter(lambda s: re.match('.+\.midi?$', s, re.I), gen)


class StatusPrinter(object):
    """Print progress from an input and an output stream at the same time"""

    def __init__(self):
        """set up the thing"""
        self._in = None
        self._out = None
        self._last_in_count = 0
        self._last_out_count = 0
        self._in_finished = False
        self._out_finished = False

    def set_input(self, in_stream):
        """wrap an input stream and return it"""
        self._in = self._wrap_stream(in_stream, 'in')
        return self._in

    def set_output(self, out_stream):
        """wrap an output stream and return it"""
        self._out = self._wrap_stream(out_stream, 'out')
        return self._out

    def _wrap_stream(self, stream, name, rate=100):
        """
        wrap a stream so that every `rate` elements we call the print function
        """
        count = 0
        for item in stream:
            count += 1
            if (count % rate) == 0:
                self._print_status(name, count)
            yield item
        self._print_status(name, count, True)

    def _print_status(self, stream, count, final=False):
        """print the status of the two streams"""
        twidth = shutil.get_terminal_size()[0]
        msg_width = min((twidth // 2) - 5, 20)
        if stream == 'in':
            self._last_in_count = count
            if final:
                self._in_finished = True
        else:
            self._last_out_count = count
            if final:
                self._out_finished = True

        in_msg = '{: <{}}'.format('in: {}'.format(self._last_in_count),
                                  msg_width)
        out_msg = '{: <{}}'.format('out: {}'.format(self._last_out_count),
                                   msg_width)
        print('\r{} -> {}'.format(in_msg, out_msg), end='', flush=True)
        if self._in_finished and self._out_finished:
            print()

    @property
    def output(self):
        return self._out

    @property
    def input(self):
        return self._in


def main(args=None):
    """Run over a bunch of files, check them out and spit them out."""
    args = parse_args(args)

    with multiprocessing.Pool(8) as pool:
        printer = StatusPrinter()
        names = generate_paths(args.input, args.recursive)
        names = printer.set_input(names)
        written = itertools.chain.from_iterable(
            pool.imap_unordered(
                partial(process_file, args.output), names, 1000))
        written = printer.set_output(written)

        unique_count, dupe_count, invalid_count = 0, 0, 0
        invalids = []
        for item in written:
            if item == '__duplicate__':
                dupe_count += 1
            elif item.startswith(args.input):
                invalids.append(item)
                invalid_count += 1
            else:
                unique_count += 1
        print('{} unique, {} duplicates, {} invalid ({} total)'.format(
            unique_count, dupe_count, invalid_count,
            invalid_count + unique_count + dupe_count))

    print('invalid files: \n{}'.format('\n'.join(invalids)))


if __name__ == '__main__':
    main()
