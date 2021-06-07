import sys
import os
import math
import pickle
import time
import pandas as pd

sys.path.append(os.path.abspath(os.path.join('')))

import audio_tools
from load.load_drums_helper import *
from paths import *

logger = logging.getLogger(__name__)


def load_drum_library(drum_lib_path):
    """
    Load all the audio files from the folder located at drum_lib_path that satisfy the required constraints
    TODO

    @param drum_lib_path: the drum library path
    @return: the name of the dataset folder, in which are stored
    - the dataset.pkl file corresponding to the resulting loaded dataframe
    - the metadata.json summarising everything
    """
    # Create the dataset folder
    folder_name, folder_path = create_dataset_folder(drum_lib_path)

    # Gather the audio files in drum_lib_path that satisfy the required constraints
    dataframe_rows, blacklisted_files, ignored_files, too_long_files, quiet_outliers = load(drum_lib_path)

    # Create the full dataframe out of the list of dictionaries, save it in the right folder, and add a metadata file
    save_all(dataframe_rows, folder_path, drum_lib_path, blacklisted_files, ignored_files, too_long_files,
             quiet_outliers)

    return folder_name


def create_dataset_folder(drum_lib_path):
    """
    TODO
    @param drum_lib_path:
    @return:
    """
    without_extra_slash = os.path.normpath(drum_lib_path)
    last_part = os.path.basename(without_extra_slash).replace(" ", "_")
    folder_name = time.strftime(f"%Y%m%d-%H%M%S-{last_part}")
    folder_path = PICKLE_DATASETS_PATH / folder_name
    os.makedirs(folder_path)
    logger.info(f'New dataset folder: {folder_name}')
    return folder_name, folder_path


def load(drum_lib_path):
    """
    TODO
    @param drum_lib_path:
    @return:
    """
    # Create empty arrays to be filled
    dataframe_rows = []
    blacklisted_files = []
    ignored_files = []
    too_long_files = []
    quiet_outliers = []

    # .wav file research loop
    logger.info(f'Searching for .wav files in {drum_lib_path}...')
    for input_file in Path(drum_lib_path).glob('**/*.wav'):

        absolute_path_name = input_file.resolve().as_posix()
        if GlobalConfig.VERBOSE:
            logger.info(absolute_path_name[len(drum_lib_path) + 1:])
        if not audio_tools.can_load_audio(absolute_path_name):
            continue

        file_stem = Path(absolute_path_name).stem.lower()
        assignment = assign_class(absolute_path_name, file_stem, blacklist_path=BLACKLIST_PATH, ignore_path=IGNORE_PATH)
        if assignment["blacklisted"] is not None:
            blacklisted_files.append(assignment["blacklisted"])
        if assignment["ignored"] is not None:
            ignored_files.append(assignment["ignored"])
        # Skip the recordings that do not belong to any of our classes
        if assignment["drum_type"] is None:
            continue

        # Initialize the properties of the current audio file
        properties = {
            'audio_path': absolute_path_name,
            'file_stem': file_stem,
            'drum_type': assignment["drum_type"],
            'start_time': 0.0,
            'end_time': np.NaN
        }

        # Load the raw audio
        raw_audio = audio_tools.load_raw_audio(absolute_path_name, fast=True)

        # Recheck readability
        if raw_audio is None:
            logger.warning(f'Skipping {absolute_path_name}, unreadable (audio is None)')
            continue

        # Add the original duration to the properties
        properties['orig_duration'] = len(raw_audio) / float(GlobalConfig.DEFAULT_SR)

        # Check for the onsets and add a new duration column
        start_time, end_time = detect_onsets(raw_audio)
        properties['start_time'] = start_time
        properties['end_time'] = end_time
        if end_time is not None and not math.isnan(end_time):
            properties["new_duration"] = float(end_time) - float(start_time)
        else:
            properties["new_duration"] = properties["orig_duration"] - start_time

        # If the new_duration attribute is bigger than MAX_SAMPLE_DURATION seconds, the audio is too long, discard it
        if properties["new_duration"] >= PreprocessingConfig.MAX_SAMPLE_DURATION:
            too_long_files.append(absolute_path_name)
            continue

        # If the audio file is too quiet, discard it
        if is_too_quiet(raw_audio):
            quiet_outliers.append(absolute_path_name)
            continue

        # Append the properties dict to the list of dictionaries, which will become a dataframe later :)
        dataframe_rows.append(properties)
    return dataframe_rows, blacklisted_files, ignored_files, too_long_files, quiet_outliers


def save_all(dataframe_rows, folder_path, input_dir_path, blacklisted_files, ignored_files, too_long_files,
             quiet_outliers):
    """

    @param dataframe_rows:
    @param folder_path:
    @param input_dir_path:
    @param blacklisted_files:
    @param ignored_files:
    @param too_long_files:
    @param quiet_outliers:
    @return:
    """
    logger.info('Saving the dataset and metadata file...')
    drums_df = pd.DataFrame(dataframe_rows)
    pickle.dump(drums_df, open(folder_path / DATASET_FILENAME, 'wb'))
    create_metadata(drums_df, input_dir_path, blacklisted_files, ignored_files, too_long_files, quiet_outliers,
                    folder_path, metadata_filename=METADATA_JSON_FILENAME)


def run_or_load(dataset_folder):
    """
    TODO
    @param dataset_folder:
    @return:
    """
    if GlobalConfig.RELOAD:
        dataset_folder = load_drum_library(GlobalConfig.SAMPLE_LIBRARY)
    if dataset_folder is not None:
        drums_df = pd.read_pickle(PICKLE_DATASETS_PATH / dataset_folder / DATASET_FILENAME)
    else:
        raise Exception('dataset_folder is None, please specify the dataset_folder name.')
    return drums_df, dataset_folder


if __name__ == "__main__":
    load_drum_library(GlobalConfig.SAMPLE_LIBRARY)
