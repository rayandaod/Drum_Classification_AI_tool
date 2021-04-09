import logging
import librosa
import pickle
import math
import pandas as pd
import numpy as np
from pathlib import Path

import params
import read_audio

logger = logging.getLogger(__name__)

DEBUG = True


def read_drum_library(input_dir_path):
    logger.info(f'Searching for audio files found in {input_dir_path}...')
    dataframe_rows = []
    for input_file in Path(input_dir_path).glob('**/*.wav'):
        absolute_path_name = input_file.resolve().as_posix()
        if DEBUG:
            print(absolute_path_name[len(input_dir_path) + 1:])
        if not read_audio.can_load_audio(absolute_path_name):
            continue

        file_stem = Path(absolute_path_name).stem.lower()
        drum_class = assign_class(absolute_path_name, file_stem)

        # Skip the recordings that do not belong to any of our classes
        if drum_class is None:
            continue

        properties = {
            'audio_path': absolute_path_name,
            'file_stem': file_stem,
            'class': drum_class,
            'start_time': 0.0,
            'end_time': np.NaN
        }
        # Tack on the original file duration (will have to load audio)
        audio = read_audio.load_raw_audio(absolute_path_name, fast=True)

        # can_load_audio check above usually catches bad files, but sometimes not
        if audio is None:
            logger.warning(f'Skipping {absolute_path_name}, unreadable (audio is None)')
            continue

        # Add the original duration to the properties
        properties['orig_duration'] = len(audio) / float(params.DEFAULT_SR)

        # Check for the onsets
        onset_dict = trim_loop(audio)
        properties['start_time'] = onset_dict["start"]
        properties['end_time'] = onset_dict["end"]

        dataframe_rows.append(properties)

    if DEBUG:
        print('     Loading files ok')

    dataframe = pd.DataFrame(dataframe_rows)

    # Add a column new_duration, by taking into account the new start and end time (after loop trimming)
    if DEBUG:
        print('Computing new durations...')
    dataframe["new_duration"] = dataframe.apply(lambda row: new_duration(row), axis=1)

    # Only keep the samples with new_duration < 5 seconds
    if DEBUG:
        print('Checking new durations...')
    len_dataframe_1 = len(dataframe)
    dataframe = dataframe[dataframe["new_duration"] <= params.MAX_SAMPLE_DURATION]
    if DEBUG:
        print("  Removed {} samples with duration > {} seconds".format(len_dataframe_1 - len(dataframe),
                                                                             params.MAX_SAMPLE_DURATION))

    # # TODO: not working properly
    # # Only keep the samples for which all the frames have an RMS above some threshold
    # if DEBUG: print('Filtering quiet outliers...')
    # len_dataframe_2 = len(dataframe)
    # dataframe = filter_quiet_outliers(dataframe)
    # if DEBUG: print("   Removed {} quiet samples".format(len_dataframe_2 - len(dataframe)))

    pickle_dataset_path = params.DATA_PATH.joinpath(params.DATAFRAME_FILENAME)
    pickle.dump(dataframe, open(pickle_dataset_path, 'wb'))
    return pickle_dataset_path.absolute().as_posix()


def assign_class(absolute_path, file_stem):
    """
    Assigns a class to the sample, excluding keywords in blacklist.
    :param absolute_path: The absolute path of the current sample
    :param file_stem: The name of the current sample
    :return: The assigned class, among those in DRUM_TYPES
    """
    for drum_type in params.DRUM_TYPES:
        # That way, we first check the file stem (in case the latter contains "hat" and the absolute path contains
        # "kick" for example)
        if drum_type in file_stem.lower() or drum_type in absolute_path.lower():

            blacklist_file = open(params.DATA_PATH / "blacklist.txt")
            for line in blacklist_file:
                blacklist = line.split(",")
                for b in blacklist:
                    if b in absolute_path.lower():
                        print("{} blacklisted".format(absolute_path))
                        return None

            ignoring_file = open(params.DATA_PATH / "ignore.txt")
            for line in ignoring_file:
                to_ignore = line.split(",")
                for ig in to_ignore:
                    if ig in absolute_path.lower():
                        absolute_path.replace(ig, "")
                        for dt in params.DRUM_TYPES:
                            if dt in file_stem.lower() or dt in absolute_path.lower():
                                return dt
            return drum_type
    return None


def trim_loop(raw_audio, sr=params.DEFAULT_SR):
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


def new_duration(row):
    if row["end_time"] is not None and not math.isnan(row["end_time"]):
        return float(row["end_time"]) - float(row["start_time"])
    else:
        return row["orig_duration"] - row["start_time"]


def filter_quiet_outliers(drum_dataframe, max_frames=params.MAX_FRAMES, max_rms_cutoff=params.MAX_RMS_CUTOFF):
    # Return a copy of the input dataframe without samples that are too quiet for a stable analysis
    # (all RMS frames < 0.02)
    def loud_enough(clip):
        raw_audio = read_audio.load_clip_audio(clip)
        frame_length = min(2048, len(raw_audio))
        S, _ = librosa.magphase(librosa.stft(y=raw_audio, n_fft=frame_length))
        rms = librosa.feature.rms(S=S, frame_length=frame_length, hop_length=frame_length // 4)[0]
        print(max(rms))
        result = max(rms) >= max_rms_cutoff
        if DEBUG and not result:
            print(clip.audio_path)
        return result

    return drum_dataframe[drum_dataframe.apply(loud_enough, axis=1)]


if __name__ == "__main__":
    pickle_path = read_drum_library("/Users/rayandaod/Documents/Prod/My_samples/KSHMR Vol. 3")
    pkl_file = open(params.DATA_PATH / params.DATAFRAME_FILENAME, 'rb')
    df = pd.read_pickle(pkl_file)
    print(df.info(verbose=True))
