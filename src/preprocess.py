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
DATA_PATH = here / '../data/'

DEBUG = True

DRUM_TYPES = ["kick", "snare", "hat", "tom"]
DEFAULT_SR = 22050  # TODO: 16k instead?
MAXIMUM_LENGTH = 5  # TODO: discard samples longer than that ?


def read_drum_library(input_dir_path):
    logger.info(f'Searching for audio files found in {input_dir_path}')

    dataframe_rows = []
    for input_file in input_dir_path.glob('**/*.wav'):
        absolute_path_name = input_file.resolve().as_posix()
        if DEBUG:
            print(absolute_path_name)
        if not can_load_audio(absolute_path_name):
            continue

        file_stem = Path(absolute_path_name).stem.lower()
        drum_class = assign_class(absolute_path_name, file_stem)
        properties = {
            'audio_path': absolute_path_name,
            'file_stem': file_stem,
            'class': drum_class,
            'start_time': 0.0,
            'end_time': np.NaN
        }
        # Tack on the original file duration (will have to load audio)
        audio = load_raw_audio(absolute_path_name, fast=True)
        if audio is None:
            logger.warning(f'Skipping {absolute_path_name}, unreadable (audio is None)')
            continue  # can_load_audio check above usually catches bad files, but sometimes not
        properties['orig_duration'] = len(audio) / float(DEFAULT_SR)

        if drum_class is not None:
            onset_dict = trim_loop(audio)
            properties['start_time'] = onset_dict["start"]
            properties['end_time'] = onset_dict["end"]

        dataframe_rows.append(properties)

    dataframe = pd.DataFrame(dataframe_rows)

    library_store_path = DATA_PATH.joinpath(DATAFRAME_FILENAME)
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
    """
    Mostly pass-through to librosa, but more defensively
    """
    try:
        time_series, sr = librosa.core.load(path_string, sr=sr, mono=True, offset=offset, duration=duration,
                                            res_type=('kaiser_fast' if fast else 'kaiser_best'))
    except (audioread.NoBackendError, audioread.DecodeError, EOFError, FileNotFoundError, ValueError, audioop.error):
        logger.warning(f'Can\'t read {path_string}')
        return None

    if (duration is None and time_series.shape[0] > 0) \
            or (duration is not None and time_series.shape[0] + 1 >= int(sr * duration)):
        return time_series
    else:
        logger.warning(f'Can\'t load {path_string} due to length, {time_series.shape[0]} {int(sr * duration)} '
                       f'{duration} {sr}')
        return None


def assign_class(absolute_path, file_stem):
    """
    Assigns a class to the sample, excluding keywords in blacklist.
    :param absolute_path: The absolute path of the current sample
    :param file_stem: The name of the current sample
    :return: The assigned class, among those in DRUM_TYPES
    """
    for drum_type in DRUM_TYPES:
        # That way, we first check the file stem (in case the latter contains "hat" and the absolute path contains
        # "kick" for example)
        if drum_type in file_stem.lower() or drum_type in absolute_path.lower():
            # blacklist_file = open(DATA_PATH/"blacklist.txt")
            # for line in blacklist_file:
            #     blacklist = line.split(",")
            #     for b in blacklist:
            #         if b in absolute_path.lower():
            #             return None
            return drum_type


def trim_loop(raw_audio, sr=DEFAULT_SR):
    """
    Finds the first onset of the sound, returns a good start time and end time that isolates the sound
    :param raw_audio: np array of audio data, from librosa.load
    :param sr: sample rate
    :return: dict with 'start' and 'end', in seconds
    """
    start = 0.0
    end = None

    # Add an empty second so that the beginning onset is recognized
    silence_to_add = 1.0
    raw_audio = np.append(np.zeros(int(silence_to_add * sr)), raw_audio)

    # Spectral flux
    hop_length = int(librosa.time_to_samples(1. / 200, sr=sr))
    onsets = librosa.onset.onset_detect(y=raw_audio, sr=sr, hop_length=hop_length, units='time')

    if len(onsets) == 0:
        return {'start': start, 'end': end}
    elif len(onsets) > 1:
        # If there are multiple onsets, cut it off just before the second one
        end = onsets[1] - (silence_to_add + 0.01)

    start = max(onsets[0] - (silence_to_add + 0.01), 0.0)
    return {'start': start, 'end': end}


if __name__ == "__main__":
    pickle_path = read_drum_library(Path("/Users/rayandaod/Documents/Prod/My_samples"))
    pkl_file = open(DATA_PATH/DATAFRAME_FILENAME, 'rb')
    df = pd.read_pickle(pkl_file)
    print(df)
