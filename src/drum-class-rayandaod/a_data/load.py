import sys
import os
import math
import pickle
import time
import pandas as pd
from os import path

sys.path.append(os.path.abspath(os.path.join('')))

from z_helpers import audio_tools
from a_data.helper import *
from z_helpers.paths import *

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
    drums_df, unreadable_files, blacklisted_files, ignored_files, too_long_files, quiet_outliers = load(drum_lib_path)

    # Create the full dataframe out of the list of dictionaries, save it in the right folder, and add a metadata file
    save_all(drums_df, folder_path, drum_lib_path, unreadable_files, blacklisted_files, ignored_files, too_long_files,
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
    folder_path = DATASETS_PATH / folder_name
    os.makedirs(folder_path)
    logger.info(f'New dataset folder: {folder_name}')
    return folder_name, folder_path


def load(some_path_or_dict, eval=False, is_sample_dict=False):
    """
    TODO
    @param path:
    @param eval:
    @return:
    """

    if is_sample_dict:
        # TODO: not clean...
        input_files = some_path_or_dict.items()
    else:
        assert path.exists(some_path_or_dict)
        input_files = Path(some_path_or_dict).glob('**/*.wav')

    # Create empty arrays to be filled
    dataframe_rows = []
    blacklisted_files = []
    ignored_files = []
    too_long_files = []

    if is_sample_dict:
        unreadable_files = dict()
        quiet_outliers = dict()
    else:
        unreadable_files = []
        quiet_outliers = []

    # .wav file research loop
    logger.info(f'Searching for .wav files...')
    for f in input_files:

        if is_sample_dict:
            dict_index = f[0]
            f = f[1]

        absolute_path_name = f.resolve().as_posix()
        if GlobalConfig.VERBOSE:
            logger.info(absolute_path_name[len(some_path_or_dict) + 1:])
        if not audio_tools.can_load_audio(absolute_path_name):
            if is_sample_dict:
                unreadable_files[str(dict_index)] = absolute_path_name
            else:
                unreadable_files.append(absolute_path_name)
            continue

        file_stem = Path(absolute_path_name).stem.lower()

        # Initialize the properties of the current audio file
        properties = {
            'audio_path': absolute_path_name,
            'file_stem': file_stem,
            'start_time': 0.0,
            'end_time': np.NaN
        }

        if not eval:
            assignment = assign_class(absolute_path_name, file_stem, blacklist_path=BLACKLIST_PATH,
                                      ignore_path=IGNORE_PATH)
            if assignment["blacklisted"] is not None:
                blacklisted_files.append(assignment["blacklisted"])
            if assignment["ignored"] is not None:
                ignored_files.append(assignment["ignored"])
            # Skip the recordings that do not belong to any of our classes
            if assignment["drum_type"] is None:
                continue
            properties['drum_type'] = assignment['drum_type']

        # Load the raw audio
        raw_audio = audio_tools.load_raw_audio(absolute_path_name, fast=True)

        # Recheck readability
        if raw_audio is None:
            logger.warning(f'Skipping {absolute_path_name}, unreadable (audio is None)')
            if is_sample_dict:
                unreadable_files[str(dict_index)] = absolute_path_name
            else:
                unreadable_files.append(absolute_path_name)
            continue

        # Add the original duration to the properties
        properties['orig_duration'] = len(raw_audio) / float(GlobalConfig.DEFAULT_SR)

        # Check for the onsets and add a new duration column
        start_time, end_time = detect_onsets(raw_audio)
        properties['start_time'] = start_time
        properties['end_time'] = end_time
        if end_time is not None and not math.isnan(end_time):
            new_duration = float(end_time) - float(start_time)
        else:
            new_duration = properties["orig_duration"] - start_time
        properties["new_duration"] = new_duration

        # If the new_duration attribute is bigger than MAX_SAMPLE_DURATION seconds, the audio is too long, discard it
        if not eval and new_duration >= PreprocessingConfig.MAX_SAMPLE_DURATION:
            too_long_files.append(absolute_path_name)
            continue

        # Load the raw audio again but this time by taking into accounts the new start and end time (thanks to onset
        # detection)
        new_raw_audio = audio_tools.load_raw_audio(absolute_path_name,
                                                   offset=start_time,
                                                   duration=new_duration,
                                                   fast=True)

        # If the audio file is too quiet, discard it
        if is_too_quiet(new_raw_audio):
            if is_sample_dict:
                quiet_outliers[str(dict_index)] = absolute_path_name
            else:
                quiet_outliers.append(absolute_path_name)
            continue

        if is_sample_dict:
            properties['dict_index'] = str(dict_index)

        # Append the properties dict to the list of dictionaries, which will become a dataframe later :)
        dataframe_rows.append(properties)
    drums_df = pd.DataFrame(dataframe_rows)
    return drums_df, unreadable_files, blacklisted_files, ignored_files, too_long_files, quiet_outliers


def save_all(drums_df, folder_path, input_dir_path, unreadible_files, blacklisted_files, ignored_files, too_long_files,
             quiet_outliers):
    """

    @param drums_df:
    @param folder_path:
    @param input_dir_path:
    @param unreadible_files:
    @param blacklisted_files:
    @param ignored_files:
    @param too_long_files:
    @param quiet_outliers:
    @return:
    """
    logger.info('Saving the dataset and metadata file...')
    pickle.dump(drums_df, open(folder_path / DATASET_FILENAME, 'wb'))
    create_metadata(drums_df, input_dir_path, unreadible_files, blacklisted_files, ignored_files, too_long_files,
                    quiet_outliers, folder_path, metadata_filename=METADATA_JSON_FILENAME)


def run_or_load(dataset_folder):
    """
    TODO
    @param dataset_folder:
    @return:
    """
    if GlobalConfig.RELOAD:
        dataset_folder = load_drum_library(GlobalConfig.SAMPLE_LIBRARY)
    if dataset_folder is not None:
        drums_df = pd.read_pickle(DATASETS_PATH / dataset_folder / DATASET_FILENAME)
    else:
        raise Exception('dataset_folder is None, please specify the dataset_folder name.')
    return drums_df, dataset_folder


if __name__ == "__main__":
    load_drum_library(GlobalConfig.SAMPLE_LIBRARY)
