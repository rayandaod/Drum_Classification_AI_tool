import pickle
import pandas as pd
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join('')))

from z_helpers import audio_tools, global_helper
from b_features.extractors import extract_features_from_single
from config import *
from z_helpers.paths import *


def extract_features_from_all(dataset_folder):
    # Extract the features from the entire dataset of the provided folder, and store them in the same folder
    drums_df_with_features, dataset_folder = load_extract_from(dataset_folder)

    # Save the new dataset with features and update the previously created metadata file into a more detailed one
    save_all(drums_df_with_features, dataset_folder)

    return drums_df_with_features


def load_extract_from(dataset_folder):
    """

    @param dataset_folder:
    @return:
    """

    def load_extract(audio_path, start_time, new_duration, drum_type):
        # Load the raw audio of the current clip/row
        # Note that load_clip_audio is used rather than load_raw_audio, in order to take into account the changes in
        # start_time, end_time, ... (due to loop trimming)
        raw_audio = audio_tools.load_raw_audio(audio_path, offset=start_time, duration=new_duration, fast=True)

        # Extract the features for a single row
        features_dict = extract_features_from_single(raw_audio, audio_path)

        # Include the row class to the features dictionary
        features_dict["drum_type"] = drum_type

        # Append the features dictionary to the list of features dictionaries in order to build a dataframe with it
        features_dict_list.append(features_dict)

    features_dict_list = []
    drums_df, dataset_folder = global_helper.load_dataset(dataset_folder)
    drums_df.apply(lambda row: load_extract(row.audio_path, row.start_time, row.new_duration, row.drum_type), axis=1)
    return pd.DataFrame(features_dict_list), dataset_folder


def save_all(drums_df_with_features, dataset_folder):
    pickle.dump(drums_df_with_features,
                open(PICKLE_DATASETS_PATH / dataset_folder / DATASET_WITH_FEATURES_FILENAME, 'wb'))
    create_metadata(drums_df_with_features, dataset_folder)


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
    """


    @param dataset_folder:
    @return:
    """
    if GlobalConfig.RELOAD:
        return extract_features_from_all(dataset_folder)
    else:
        return pd.read_pickle(PICKLE_DATASETS_PATH / dataset_folder / DATASET_WITH_FEATURES_FILENAME)


if __name__ == "__main__":
    extract_features_from_all(global_helper.parse_args(global_helper.global_parser()).folder)
