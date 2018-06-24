"""
Load some embeddings and do a random walk through them.
"""
import itertools
import os
import sys
import time
from functools import lru_cache
from itertools import chain

import librosa
import numpy as np
import pyaudio
from sklearn.neighbors import kneighbors_graph

from organise import read_embeddings
from player import audio_output


# ----------------------------------------------------------------
# NOTE: may be moved into the main sample player when it exists
@lru_cache()
def load_sample(path):
    """Load audio file at `path`."""
    return librosa.load(path, sr=44100)[0]


# -----------------------------------------------------------------


def main():
    """quick test"""
    embeddings = read_embeddings(sys.argv[1])

    samples = (load_sample(os.path.join(os.path.dirname(sys.argv[1]), path))
               for path in embeddings[0])
    with audio_output(
            1024, format=pyaudio.get_format_from_width(4),
            rate=44100) as output_queue:
        for sample in samples:
            output_queue.put(sample)
            time.sleep(max(0, sample.shape[0] / 44100 - 2))


if __name__ == '__main__':
    main()
