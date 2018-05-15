"""
Convert a big pile of drum midi files into a really big JSON file.
"""
import argparse
import json
import multiprocessing
import os
import re

import mido

from drums import data


def parse_args(args=None):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--output', '-o', help='file to output to')
    parser.add_argument(
        'input_dir',
        help='path of a directory containing midi files to convert')

    return parser.parse_args(args)


def find_files(directory, pattern='.*\.mid$'):
    """generate all the filenames in a directory"""
    pat = re.compile(pattern)
    return (os.path.join(directory, fname) for fname in os.listdir(directory)
            if re.match(pattern, fname))


def load_and_encode(fname):
    """Load a midi file, encode it and return the encoded data and the original
    filename in a way that it can be written easily to JSON."""
    mfile = mido.MidiFile(fname)
    encoded_track = data.encode_drum_track(mfile)
    return {'file': os.path.basename(fname), 'data': encoded_track}


def print_count(gen, num, print_every=1000):
    """yield items from a generator, printing a periodic count as we go"""
    print('0/{}'.format(num), end='', flush=True)
    for i, item in enumerate(gen):
        yield item
        if ((i + 1) % print_every) == 0:
            print('\r{}/{}'.format(i + 1, num), end='', flush=True)
    print()


def main(args=None):
    """Find all the midi files, convert them, get ready to write."""
    args = parse_args(args)
    filenames = list(find_files(args.input_dir))
    print('found {} files to process'.format(len(filenames)))

    with multiprocessing.Pool(4) as pool:
        results = pool.imap_unordered(
            load_and_encode, filenames, chunksize=1000)
        results = print_count(results, len(filenames))
        print('all files processed, writing')
        with open(args.output, 'w') as outfile:
            json.dump({'tracks': list(results)}, outfile)
    print('bye')


if __name__ == '__main__':
    main()
