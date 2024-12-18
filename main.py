import itertools
import pprint
import re
import time

from collections import deque
from math import floor

import numpy as np
import pyaudio
import sox
from queue import Queue
from threading import Thread, Condition
import soundfile

import logging

from pywhispercpp.model import Model

CHUNK = 1024
RATE = 48000
PROCESS_SIZE = 96
MODEL = 'base'
FILEPATH = 'test2.wav'
LOG_LEVEL = logging.INFO #logging.DEBUG
LIVE_FEED = False
WINDOWS = False
WINDOWS_RETAIN = 0.3
CONTEXT_RETAIN = True
BANNED_WORDS = {
    "szklanka",
    "niepożądanych"
}

all_text = []

queue = Queue()
consumer_ready = Condition()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(relativeCreated)dms - %(levelname)s - %(message)s",
    force=True
)

logging.getLogger().setLevel(LOG_LEVEL)

def check_for_banned_words(string):
    words = re.sub(r'[^\w\s]', '', string)
    words = words.split()
    words = [word.lower() for word in words]
    words = set(words)
    string = string.lower()

    logging.debug(f"words = {pprint.pformat(words)}")

    if len(words & BANNED_WORDS) > 0:
        logging.error(f"Wykryto niedozwolone słowa w segmencie '{string}'!")

def iter_chunks(iterable, chunk_size):
    iterator = iter(iterable)
    while True:
        chunk = list(itertools.islice(iterator, chunk_size))
        if not chunk:
            break
        yield chunk

def consumer(queue):
    tr = sox.Transformer()
    tr.set_input_format(file_type='wavpcm', rate=48000, bits=32, channels=1, encoding="floating-point")
    tr.set_output_format(file_type='wavpcm', rate=16000, bits=32, channels=1, encoding="floating-point")
    windowed_audio = deque(maxlen=PROCESS_SIZE)
    model = Model(
        model=MODEL,
        n_threads=8,
        single_segment=True,
        language="pl",
        no_context=CONTEXT_RETAIN if WINDOWS else True,
        print_realtime=False,
        print_timestamps=False,
        log_level=logging.getLogger().getEffectiveLevel(),
    )
    with consumer_ready:
        consumer_ready.notify()

    while True:
        item = queue.get()

        if item is None:
            break

        windowed_audio.append(item)
        if len(windowed_audio) >= PROCESS_SIZE:
            audio = np.concat(windowed_audio, axis=None)
            audio = audio.reshape((audio.size, 1))
            audio = tr.build_array(input_array=audio, sample_rate_in=48000)

            segments = model.transcribe(
                media=audio,
            )

            for i, segment in enumerate(segments):
                logging.info(f'spk {i}> {segment.text}')
                check_for_banned_words(segment.text)
                all_text.append(segment.text)

            if WINDOWS:
                [windowed_audio.popleft() for _ in range(len(windowed_audio) - floor(len(windowed_audio) * WINDOWS_RETAIN))]
            else:
                windowed_audio.clear()
            queue.task_done()
    del model
    logging.info("Bye from consumer!")

consumer_thread = Thread(target=consumer, args=(queue,))
consumer_thread.start()

logging.info('Waiting for consumer thread to bootstrap...')

with consumer_ready:
    consumer_ready.wait()

p = pyaudio.PyAudio()

stream = p.open(
    format=p.get_format_from_width(4),
    channels=1,
    rate=RATE,
    input=LIVE_FEED,
    output=True,
    frames_per_buffer=CHUNK,
)

logging.info('Listening..., press ^C to stop')

if LIVE_FEED:
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)

            byte_array = np.frombuffer(data[:], dtype=np.float32).copy()

            queue.put_nowait(byte_array)
            stream.write(data)
    except KeyboardInterrupt:
        logging.info('Stopping...')
        stream.close()
        p.terminate()
else:
    with soundfile.SoundFile(
        file=FILEPATH,
        mode='rb',
    ) as wf:
        bytesdata = wf.read(dtype="float32")
        if np.ndim(bytesdata) > 1:
            bytesdata = bytesdata[:, 0] #extract first channel only
        try:
            # for data in itertools.cycle(iter_chunks(bytesdata, CHUNK)):
            start_time = time.time()
            logging.info('Started processing')
            for data in iter_chunks(bytesdata, CHUNK):
                byte_array = np.fromiter(data[:], dtype=np.float32).copy()
                queue.put_nowait(byte_array)
                # stream.write(byte_array.tobytes())
        except KeyboardInterrupt:
            logging.info('Stopping...')
            stream.close()
            p.terminate()

queue.put(None)
consumer_thread.join()

elapsed_time = time.time() - start_time
logging.info(f'Processing ended, took {elapsed_time:.6f} seconds')

logging.info(f'Matched text: {"".join(all_text)}')

logging.info('Bye!')
