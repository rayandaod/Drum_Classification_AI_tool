import logging
import os
import librosa
import audioread
import pickle
import audioop
import pandas as pd
import numpy as np


from pathlib import Path

logger = logging.getLogger(__name__)
DATAFRAME_FILENAME = 'dataset.pkl'
here = Path(__file__).parent
INTERIM_PATH = here/'../data/interim'

DEFAULT_SR = 22050  # TODO: 16k instead?


def read_drum_library(input_dir_path):
    logger.info(f'Searching for audio files found in {input_dir_path}')

    dataframe_rows = []
    for input_file in input_dir_path.glob('**/*.*'):
        absolute_path_name = input_file.resolve().as_posix()
        if not can_load_audio(absolute_path_name):
            continue

        properties = {
            'audio_path': absolute_path_name,
            'file_stem': Path(absolute_path_name).stem.lower(),
            'start_time': 0.0,
            'end_time': np.NaN
        }
        # Tack on the original file duration (will have to load audio)
        audio = load_raw_audio(absolute_path_name, fast=True)
        if audio is None:
            continue  # can_load_audio check above usually catches bad files, but sometimes not
        properties['orig_duration'] = len(audio) / float(DEFAULT_SR)

        dataframe_rows.append(properties)

    dataframe = pd.DataFrame(dataframe_rows)

    pickle.dump(dataframe, open(library_store_path, 'wb'))
    return library_store_path.absolute().as_posix()


def can_load_audio(path_string):
    if not os.path.isfile(path_string):
        return False
    try:
        librosa.core.load(path_string, mono=True, res_type='kaiser_fast', duration=.01)
    except (audioread.NoBackendError, audioread.DecodeError, EOFError, FileNotFoundError, ValueError, audioop.error):
        logger.warning(f'Skipping {path_string}, unreadable')
        return False
    return True


def load_raw_audio(path_string, sr=DEFAULT_SR, offset=0, duration=None, fast=False):
    '''
    Mostly pass-through to librosa, but more defensively
    '''
    try:
        time_series, sr = librosa.core.load(path_string, sr=sr, mono=True, offset=offset, duration=duration,
                                            res_type=('kaiser_fast' if fast else 'kaiser_best'))
    except (audioread.NoBackendError, audioread.DecodeError, EOFError, FileNotFoundError, ValueError, audioop.error):
        logger.warning(f'Can\'t read {path_string}')
        return None

    if (duration is None and time_series.shape[0] > 0)\
            or (duration is not None and time_series.shape[0] + 1 >= int(sr * duration)):
        return time_series
    else:
        logger.warning(f'Can\'t load {path_string} due to length, {time_series.shape[0]} {int(sr * duration)} {duration} {sr}')
        return None