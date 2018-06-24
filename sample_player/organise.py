"""
Learn an encoding of a bunch of short samples so that we can do a random walk
in the learnt latent space and find perceptually similar samples.

The goal isn't to be able to smoothly interpolate embeddings and generate
new samples so we'll steer clear of computationally heavy methods like
a wavenet autoencoder etc. Ideally we just want to do the analysis once and
produce an index we can use to do a nearest neighbour lookup later on as
we run around the sample space. If we want to get fancy we can always just
literally blend nearby samples.

We will make a fixed size data representation by pooling the spectrograms,
we are more concerned with the overall spectral content than the envelopes.
This is because we are currently mostly concerned with drum samples which are
all pretty similar wrt. their temporal behaviour.

Once the data is ready we need to find an embedding. There's a lot of ways
to do this -- three relatively straightforward methods are:
- T-SNE or UMAP: manifold learning methods that will definitely find reasonable
  embeddings for each data point but which don't really stream so will only
  work if the processed data comes out small enough. Probably won't scale to
  more than a few hundred thousand samples, although it's unlikely we'll put
  together more than that.
- VAE: Learns a nice latent space because it allows us to specify a prior over
  it's distribution which is typically a convenient diagonal Gaussian. This
  would make it nice and easy to do a random walk without skipping over weird
  patterns in a embedding manifold because it should stay relatively smooth.
- Triplet loss / siamese network style embeddings: fewer guarantees on the
  embedded space potentially but often easier to train than a VAE. Might
  require a more sophisticated data pipeline with augmentations to be able to
  generate positive associations to train against but I've had good experiences
  with this because you can get away with pretty small models and you don't
  have to waste time making reconstructions that we don't need to keep ever.
"""
import argparse
import itertools
import multiprocessing
import os
import re
import time
from functools import partial

import librosa
import librosa.feature
import numpy as np
import umap


def generate_sample_paths(directory, recurse=True):
    """
    Find (potentially recursively) all the audio files we can load in
    `directory`.

    Args:
        directory (str): path to a directory in which to look for samples.
        recurse (Optional[bool]): whether to search recursively or not.
            Default is True, indicating a recursive search.

    Yields:
        str: paths, one at a time.
    """
    audio_pattern = re.compile(
        '.+\.(wav)|(mp4)|(ogg)|(mp4)|(aiff?)', flags=re.I)

    if recurse:
        gen = itertools.chain.from_iterable(
            map(lambda x: map(partial(os.path.join, x[0]), x[2]),
                os.walk(directory)))
    else:
        gen = filter(os.path.isfile,
                     map(
                         partial(os.path.join, directory),
                         os.listdir(directory)))
    return filter(lambda s: re.match(audio_pattern, s), gen)


def _extract_features(fname, sample_rate, n_fft, hop_size, n_bins):
    """Extract a single set of features."""
    audio, _ = librosa.load(fname, sr=sample_rate)
    if audio.shape[0] == 0:
        print('{} is empty :|'.format(fname))
        return None
    mel_specgram = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_size,
        power=2,
        n_mels=n_bins,
        fmin=20.0,
        fmax=sample_rate / 2.5)

    return os.path.abspath(fname), np.log(1e-7 + np.mean(mel_specgram, 1))


def extract_audio_features(audio_files,
                           sample_rate=44100.,
                           n_fft=1024,
                           hop_size=256,
                           n_bins=128):
    """Preprocess the audio into pooled mel-spectrograms.

    Args:
        audio_files: iterator producing paths to audio files.
        sample_rate: assumed sample rate. Audio may be resampled to this.
        n_fft: fft size for the stft prior to pooling and mel-filtering.
        hop_size: hop size of the stft prior to pooling and mel-filtering.
        n_bins: number of mel bins, this is the final dimensionality of the
            output.

    Yields:
        filename: the original filename
        numpy array: the embedded audio
    """
    with multiprocessing.Pool(4) as pool:
        processor = partial(
            _extract_features,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_size=hop_size,
            n_bins=n_bins)
        results = filter(None, pool.imap_unordered(processor, audio_files,
                                                   100))
        for result in results:
            yield result


def write_embeddings(embedding_generator, path):
    """Write all the embedding pairs into a simple tsv file with no headers in
    which the first column is the name of the original and the rest is the
    each dimension of the embedding.
    """
    with open(path, 'w') as outfile:
        for name, embedding in embedding_generator:
            # make the name relative to the directory we're writing into
            fname = os.path.relpath(name, os.path.dirname(path))
            fname = fname.replace('\t', ' ')
            row = '{}\t{}\n'.format(fname,
                                    '\t'.join([str(dim) for dim in embedding]))
            outfile.write(row)
        outfile.write('\n')


def _read_row(row):
    """read a single row of embeddings"""
    name, *emb = row.split('\t')
    if len(emb) == 0:
        return None
    emb = np.array([float(dim.strip()) for dim in emb])
    return name, emb


def read_embeddings(embedding_file):
    """Read a file of embeddings in as a numpy array and a list of names"""
    with open(embedding_file) as infile:
        names, embeddings = zip(*filter(None, map(_read_row, infile)))
    return names, np.stack(embeddings)


def _remove_duplicates(filenames, embeddings, threshold=1):
    """
    Attempt to remove embeddings that are very near each other.
    Potentially involves a lot of distance calculations but there's no other
    great way.
    """
    seen = []
    for fname, embedding in zip(filenames, embeddings):
        # slightly roundabout so we can print what is getting skipped for QA
        if seen:
            nearest = min(seen, key=lambda x: np.sum(np.abs(embedding - x[1])))
            dist = np.sum(np.abs(embedding - nearest[1]))
            if dist < threshold:
                print('{} == {}  ({:.4f}), skipping'.format(
                    fname, nearest[0], dist))
            else:
                seen.append((fname, embedding))
                yield fname, embedding
        else:
            seen.append((fname, embedding))


def umap_dimensionality_reduction(embeddings, new_dims):
    """reduce the dimensionality of the data with UMAP"""
    return umap.UMAP(n_components=new_dims).fit_transform(embeddings)


def parse_args(args=None):
    """parse a couple of arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', '-i', help='directory of files to encode')
    parser.add_argument(
        '--recursive',
        '-r',
        action='store_true',
        help='whether or not to recurse into subdirectories')
    parser.add_argument(
        '--mel_embeddings',
        '-m',
        help='where to look for  mel embeddings, '
        'if not there they will be generated')
    parser.add_argument(
        '--final_size',
        '-s',
        type=int,
        default=16,
        help='size of the final processed embeddings')
    parser.add_argument(
        '--output', '-o', help='path for the final embedding output')
    return parser.parse_args(args)


def main(args=None):
    """Encode some data and write it out."""
    args = parse_args(args)

    if not os.path.isfile(args.mel_embeddings):
        print('Computing mel features')
        mel_embeddings = list(
            extract_audio_features(
                generate_sample_paths(args.input, args.recursive)))
        # remove any that are _very_ close to each other because there are
        # some duplicates in the dataset.
        print('removing duplicates')
        mel_embeddings = list(_remove_duplicates(*zip(*mel_embeddings)))
        write_embeddings(mel_embeddings, args.mel_embeddings)
        names, mel_embeddings = zip(*mel_embeddings)
        mel_embeddings = np.stack(mel_embeddings)
    else:
        print('Reading mel features')
        names, mel_embeddings = read_embeddings(args.mel_embeddings)

    print('using UMAP to reduce from {} to {} dimensions'.format(
        mel_embeddings.shape[1], args.final_size))
    start = time.time()
    final_embeddings = umap_dimensionality_reduction(mel_embeddings,
                                                     args.final_size)
    end = time.time()
    print('UMAP finished in {:.4f}s'.format(end - start))
    print('writing UMAP embeddings')
    write_embeddings(zip(names, final_embeddings), args.output)


if __name__ == '__main__':
    main()
