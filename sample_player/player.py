"""
Play samples in the background kind of.
"""

import contextlib
import queue
import threading
import time

import numpy as np
import pyaudio

STOP_AUDIO = 'stop it!'


def _sum_bus(buffer, additions):
    """add all the elements in `additions` to `buffer`"""
    alive = []
    for item in additions:
        if item.shape[0] < buffer.shape[0]:
            buffer[:item.shape[0]] += item
        elif item.shape[0] > buffer.shape[0]:
            buffer += item[:buffer.shape[0]]
            alive.append(item[buffer.shape[0]:])
        else:
            buffer += item
    return alive


def _make_callback(in_buffer, out_buffer):
    """make a callback that will cycle through `in_buffer`, zeroing it out as
    it goes."""

    def _callback(_, frame_count, __, ___):
        """actually push samples"""
        if not hasattr(_callback, 'pos'):
            _callback.pos = 0
        out_buffer[:frame_count] = in_buffer[_callback.pos:
                                             _callback.pos + frame_count]
        in_buffer[_callback.pos:_callback.pos + frame_count] = 0
        _callback.pos = (_callback.pos + frame_count) % in_buffer.shape[0]
        return out_buffer[:frame_count], pyaudio.paContinue

    return _callback


def _play_audio(audio_queue, chunk_size, **kwargs):
    """
    Pull samples off `queue` and play them.
    """
    local_data = threading.local()
    local_data.buffer = np.zeros(
        chunk_size * 20,
        dtype=np.float32)  # avoid allocations while we're processing
    local_data.pending = []
    local_data.bus = np.zeros(chunk_size * 2, dtype=np.float32)

    pa_obj = pyaudio.PyAudio()
    stream = pa_obj.open(
        output=True,
        stream_callback=_make_callback(local_data.buffer,
                                       np.zeros(chunk_size, dtype=np.float32)),
        **kwargs)

    pos = 0
    while True:
        # local_data.buffer[:] = 0
        try:
            new_samples = audio_queue.get_nowait()
            if new_samples is STOP_AUDIO:
                break
            local_data.pending.append(new_samples)
        except queue.Empty:
            pass
        if local_data.pending:
            local_data.bus[:] = 0
            local_data.pending = _sum_bus(local_data.bus, local_data.pending)
            local_data.buffer[pos:
                              pos + local_data.bus.shape[0]] = local_data.bus
            pos = (pos + local_data.bus.shape[0]) % local_data.buffer.shape[0]
            time.sleep(local_data.bus.shape[0] / kwargs['rate'])


@contextlib.contextmanager
def audio_output(chunk_size, **kwargs):
    """
    Context manager which opens a pyaudio stream in the background and returns
    a queue to drop audio samples on it.

    Args:
        all args are just passed on to the pyaudio output stream (except that
        it will always be opened with `output=True`).

    Yields:
        queue: a queue expecting numpy arrays of samples which will get summed
            into the output starting as soon as possible.
    """
    pa_obj = pyaudio.PyAudio()
    if 'channels' not in kwargs:
        kwargs['channels'] = 1
    audio_queue = queue.Queue()
    audio_thread = threading.Thread(
        target=_play_audio,
        kwargs=dict(audio_queue=audio_queue, chunk_size=chunk_size, **kwargs))
    audio_thread.start()

    yield audio_queue

    audio_queue.put(STOP_AUDIO)
    audio_thread.join()
