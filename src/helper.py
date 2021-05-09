import os
import warnings
import pickle
import logging
import pandas as pd
from os import path
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from argparse import ArgumentParser

from config import TrainingConfig, PathConfig, GlobalConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_global_parser():
    parser = ArgumentParser()
    parser.add_argument('--reload', action='store_true', dest='reload',
                        help='Reload the drum library and extract the features again.')
    parser.add_argument('--old', type=str, default=None, help='Select an already loaded dataset')
    parser.add_argument('--verbose', action='store_true', dest='verbose',
                        help='Print useful data for debugging')
    parser.set_defaults(reload=False)
    parser.set_defaults(verbose=False)
    return parser


def parse_global_arguments(parser):
    args = parser.parse_args()

    GlobalConfig.RELOAD = args.reload
    logger.info(f"Reload = {GlobalConfig.RELOAD}")

    GlobalConfig.VERBOSE = args.verbose
    logger.info(f"Verbose = {GlobalConfig.VERBOSE}")

    return args


# Check that the given file path does not already exists.
# If it does, add "_new" at the end (before the extension)
# If it doesn't, simply return the originally given path
def can_write(file_path):
    if path.exists(file_path):
        warnings.warn("The given path already exists, adding \"_new\" at the end.")
        path_split = os.path.splitext(file_path)
        return can_write(path_split[0] + "_new" + path_split[1])
    else:
        return file_path


def prepare_data(drums_df):
    drums_df_caped = drums_df.groupby('drum_type').head(TrainingConfig.N_SAMPLES_PER_CLASS)
    drum_type_labels, unique_labels = pd.factorize(drums_df_caped.drum_type)
    drums_df_labeled = drums_df_caped.assign(drum_type_labels=drum_type_labels)

    train_clips_df, val_clips_df = train_test_split(drums_df_labeled, random_state=GlobalConfig.RANDOM_STATE)
    logger.info(f'{len(train_clips_df)} training samples, {len(val_clips_df)} validation samples')

    # Remove last two columns (drum_type_labels and drum_type) for training
    columns_to_drop = ['drum_type_labels', 'drum_type']
    train_np = drop_columns(train_clips_df, columns_to_drop)
    test_np = drop_columns(val_clips_df, columns_to_drop)

    # There are occasionally random gaps in descriptors, so use imputation to fill in all values
    train_np, test_np = imputer(train_np, test_np)

    # Standardize features by removing the mean and scaling to unit variance
    train_np, test_np = scaler(train_np, test_np)

    return train_np, train_clips_df.drum_type_labels, test_np, val_clips_df.drum_type_labels, list(unique_labels.values)


def drop_columns(df, columns):
    for col in columns:
        df = df.drop(labels=col, axis=1)
    return df.to_numpy()


# Use imputation to fill in all missing values
def imputer(train_np, test_np):
    try:
        imp = pickle.load(open(PathConfig.IMPUTATER_PATH, 'rb'))
    except FileNotFoundError:
        logger.info(f'No cached imputer found, training')
        imp = TrainingConfig.iterative_imputer
        imp.fit(train_np)
        pickle.dump(imp, open(PathConfig.IMPUTATER_PATH, 'wb'))
    train_np = imp.transform(train_np)
    test_np = imp.transform(test_np)
    return train_np, test_np


# Standardize the training and testing sets by removing the mean and scaling to unit variance
def scaler(train_np, test_np):
    scaler = preprocessing.StandardScaler().fit(train_np)
    train_np = scaler.transform(train_np)
    test_np = scaler.transform(test_np)
    pickle.dump(scaler, open(PathConfig.SCALER_PATH, 'wb'))
    return train_np, test_np
