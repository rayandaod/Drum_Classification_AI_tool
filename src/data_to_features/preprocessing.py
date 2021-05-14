import librosa
import pickle
import math
import pandas as pd
import time
import os
import json

import data_to_features.helper as helper
from config import *

logger = logging.getLogger(__name__)


def read_drum_library(input_dir_path):
    logger.info(f'Searching for audio files found in {input_dir_path}...')
    dataframe_rows = []

    # Create the dataset folder
    folder_name = time.strftime("%Y%m%d-%H%M%S")
    folder_path = PICKLE_DATASETS_PATH / folder_name
    os.makedirs(folder_path)

    for input_file in Path(input_dir_path).glob('**/*.wav'):
        absolute_path_name = input_file.resolve().as_posix()
        if GlobalConfig.VERBOSE:
            print(absolute_path_name[len(input_dir_path) + 1:])
        if not helper.can_load_audio(absolute_path_name):
            continue

        file_stem = Path(absolute_path_name).stem.lower()
        drum_class = assign_class(absolute_path_name, file_stem, folder_name)

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
        audio = helper.load_raw_audio(absolute_path_name, fast=True)

        # can_load_audio check above usually catches bad files, but sometimes not
        if audio is None:
            logger.warning(f'Skipping {absolute_path_name}, unreadable (audio is None)')
            continue

        # Add the original duration to the properties
        properties['orig_duration'] = len(audio) / float(GlobalConfig.DEFAULT_SR)

        # Check for the onsets
        onset_dict = detect_onsets(audio)
        properties['start_time'] = onset_dict["start"]
        properties['end_time'] = onset_dict["end"]

        dataframe_rows.append(properties)

    if GlobalConfig.VERBOSE:
        print('     Loading files ok')

    dataframe = pd.DataFrame(dataframe_rows)

    # Add a column new_duration, by taking into account the new start and end time (after loop trimming)
    if GlobalConfig.VERBOSE:
        print('Computing new durations...')
    dataframe["new_duration"] = dataframe.apply(lambda row: new_duration(row), axis=1)

    # Only keep the samples with new_duration < 5 seconds
    if GlobalConfig.VERBOSE:
        print('Checking new durations...')
    len_dataframe_1 = len(dataframe)
    dataframe = dataframe[dataframe["new_duration"] <= PreprocessingConfig.MAX_SAMPLE_DURATION]
    if GlobalConfig.VERBOSE:
        print(" Removed {} samples with duration > {} seconds".format(len_dataframe_1 - len(dataframe),
                                                                      PreprocessingConfig.MAX_SAMPLE_DURATION))

    # Only keep the samples for which all the frames have an RMS above some threshold
    if GlobalConfig.VERBOSE:
        print('Filtering quiet outliers...')
    len_dataframe_2 = len(dataframe)
    dataframe = filter_quiet_outliers(dataframe, folder_name)
    if GlobalConfig.VERBOSE:
        print(" Removed {} quiet samples".format(len_dataframe_2 - len(dataframe)))

    # Save the pickle file in the right folder
    pickle.dump(dataframe, open(folder_path / "dataset.pkl", 'wb'))

    # Create the metadata.json
    metadata = {
        "n_samples": str(dataframe.size),
        "classes": {}
    }
    for drum_type in GlobalConfig.DRUM_TYPES:
        metadata["classes"][drum_type] = str(dataframe[dataframe["class"] == drum_type].size)
    with open(folder_path / METADATA_JSON_FILENAME, 'w') as outfile:
        json.dump(metadata, outfile)

    return folder_name


def assign_class(absolute_path, file_stem, dataset_folder):
    """
    Assigns a class to the sample, excluding keywords in blacklist.
    :param absolute_path: The absolute path of the current sample
    :param file_stem: The name of the current sample
    :return: The assigned class, among those in DRUM_TYPES
    """
    for drum_type in GlobalConfig.DRUM_TYPES:
        # That way, we first check the file stem (in case the latter contains "hat" and the absolute path contains
        # "kick" for example)
        if drum_type in file_stem.lower() or drum_type in absolute_path.lower():

            # TODO: not working
            with open(PICKLE_DATASETS_PATH / dataset_folder / BLACKLISTED_FILES_FILENAME, "w") \
                    as blacklisted_files:
                blacklist_file = open(BLACKLIST_PATH)
                for line in blacklist_file:
                    blacklist = line.split(",")
                    for b in blacklist:
                        if b in absolute_path.lower():
                            blacklisted_files.write("\n{}".format(absolute_path))
                            return None

            ignoring_file = open(IGNORE_PATH)
            for line in ignoring_file:
                to_ignore = line.split(",")
                for ig in to_ignore:
                    if ig in absolute_path.lower():
                        absolute_path.replace(ig, "")
                        for dt in GlobalConfig.DRUM_TYPES:
                            if dt in file_stem.lower() or dt in absolute_path.lower():
                                return dt
            return drum_type
    return None


def detect_onsets(raw_audio, sr=GlobalConfig.DEFAULT_SR):
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
    hop_length = int(sr * PreprocessingConfig.SR_FRACTION_FOR_TRIM)
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


def filter_quiet_outliers(drum_dataframe, dataset_folder, max_frames=PreprocessingConfig.MAX_FRAMES,
                          max_rms_cutoff=PreprocessingConfig.MAX_RMS_CUTOFF, verbose=False):
    # Return a copy of the input dataframe without samples that are too quiet for a stable analysis
    # (RMS < 0.02 for all frames up to PreprocessingConfig.MAX_FRAMES (approximately 1 second))

    def loud_enough(clip):
        raw_audio = helper.load_clip_audio(clip)
        frame_length = min(2048, len(raw_audio))
        S, _ = librosa.magphase(librosa.stft(y=raw_audio, n_fft=frame_length))
        rms = librosa.feature.rms(S=S, frame_length=frame_length, hop_length=frame_length // 4)[0]
        result = max(rms[:max_frames]) >= max_rms_cutoff
        if not result:
            quiet_outliers_file.write("\n{}".format(clip.audio_path))
            if verbose:
                print(clip.audio_path)
        return result

    with open(PICKLE_DATASETS_PATH / dataset_folder / QUIET_OUTLIERS_FILENAME, 'w') as \
            quiet_outliers_file:
        df = drum_dataframe[drum_dataframe.apply(loud_enough, axis=1)]
    return df


def load_drums_df(dataset_folder):
    if GlobalConfig.RELOAD:
        dataset_folder = read_drum_library(GlobalConfig.SAMPLE_LIBRARY)
    if dataset_folder is not None:
        drums_df = pd.read_pickle(PICKLE_DATASETS_PATH / dataset_folder / DATASET_FILENAME)
    else:
        raise Exception('dataset_folder is None, please specify the dataset_folder name.')
    return drums_df, dataset_folder


if __name__ == "__main__":
    parser = global_parser()
    args = parse_args(parser)
    load_drums_df(args.old)
