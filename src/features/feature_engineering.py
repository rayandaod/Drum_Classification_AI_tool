import pickle
import pandas as pd
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join('..')))

import audio_tools
import features.feature_helper as feature_helper
from features.feature_extractors import extract_features_from_single
from config import *


def extract_features_from_all(dataset_folder):
    # If dataset_folder is not given, take the last created dataset folder, otherwise load_drums it from the
    # provided folder
    drums_df = load_dataset(dataset_folder)

    # Extract the features from the entire dataset of the provided folder, and store them in the same folder
    drums_df_with_features = extract_and_store(drums_df, dataset_folder)

    # Update the previously created metadata file into a more detailed one
    create_metadata(drums_df_with_features, dataset_folder)

    return drums_df_with_features


def load_dataset(dataset_folder):
    """

    @param dataset_folder:
    @return:
    """
    if dataset_folder is None:
        all_subdirs = feature_helper.all_subdirs_of(PICKLE_DATASETS_PATH)
        dataset_folder = max(all_subdirs, key=os.path.getmtime)
    return pd.read_pickle(PICKLE_DATASETS_PATH / dataset_folder / DATASET_FILENAME)


def extract_and_store(drums_df, dataset_folder):
    """

    @param drums_df:
    @param dataset_folder:
    @return:
    """

    def extract_lamdba(clip):
        # Load the raw audio of the current clip/row
        # Note that load_clip_audio is used rather than load_raw_audio, in order to take into account the changes in
        # start_time, end_time, ... (due to loop trimming)
        raw_audio = audio_tools.load_clip_audio(clip)

        # Extract the features for a single row
        features_dict = extract_features_from_single(raw_audio)

        # Include the row class to the features dictionary
        features_dict["drum_type"] = clip["drum_type"]

        # Append the features dictionary to the list of features dictionaries in order to build a dataframe with it
        features_dict_list.append(features_dict)

    features_dict_list = []
    drums_df.apply(lambda row: extract_lamdba(row), axis=1)
    drums_df_with_features = pd.DataFrame(features_dict_list)
    pickle.dump(drums_df_with_features,
                open(PICKLE_DATASETS_PATH / dataset_folder / DATASET_WITH_FEATURES_FILENAME, 'wb'))
    return drums_df_with_features


def create_metadata(drums_df_with_features, dataset_folder):
    """
    Retrieve the previous metadata.json file and complete it with the extracted features

    @param drums_df_with_features:
    @param dataset_folder:
    @return:
    """
    # Retrieve the column names of the feature dataframe
    # Doing the following otherwise drums_df_with_features.columns is not serializable
    columns = []
    for col_name in drums_df_with_features.columns:
        columns.append(col_name)

    # Load the previously created metadata file
    metadata_path = PICKLE_DATASETS_PATH / dataset_folder / METADATA_JSON_FILENAME
    with open(metadata_path, "r+") as metadata_file:
        metadata_dict = json.load(metadata_file)

    # Add feature related info to the metadata dictionary
    metadata_dict['n_columns'] = len(columns)
    metadata_dict["columns"] = columns

    # Save the metadata file
    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata_dict, metadata_file)


def run_or_load(dataset_folder):
    if GlobalConfig.RELOAD:
        return extract_features_from_all(dataset_folder)
    else:
        return pd.read_pickle(PICKLE_DATASETS_PATH / dataset_folder / DATASET_FILENAME)


if __name__ == "__main__":
    extract_features_from_all(parse_args(global_parser()).folder)
