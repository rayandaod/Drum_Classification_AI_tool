import os
import pandas as pd
from argparse import ArgumentParser

from config import *
from z_helpers.paths import *


def global_parser():
    parser = ArgumentParser()
    parser.add_argument('--reload', action='store_true', dest='reload',
                        help='Reload the drum library and extract the features again.')
    parser.add_argument('--folder', type=str, default=None, help='Select an already loaded dataset')
    parser.add_argument('--verbose', action='store_true', dest='verbose',
                        help='Print useful data for debugging')
    parser.set_defaults(reload=False)
    parser.set_defaults(verbose=False)
    return parser


def parse_args(parser):
    args = parser.parse_args()

    GlobalConfig.RELOAD = args.reload
    logger.info(f"Reload = {GlobalConfig.RELOAD}")

    GlobalConfig.VERBOSE = args.verbose
    logger.info(f"Verbose = {GlobalConfig.VERBOSE}")

    return args


def load_dataset(dataset_folder, dataset_filename=DATASET_FILENAME):
    """


    @param dataset_folder:
    @param dataset_filename:
    @return:
    """
    if dataset_folder is None:
        all_subdirs = all_subdirs_of(PICKLE_DATASETS_PATH)
        dataset_folder = max(all_subdirs, key=os.path.getmtime)
    drums_df = pd.read_pickle(PICKLE_DATASETS_PATH / dataset_folder / dataset_filename)
    return drums_df, dataset_folder


def all_subdirs_of(b='.'):
    """


    @param b:
    @return:
    """
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd): result.append(bd)
    return result


# Retrieve a minimised version of a file path, with the slashes ("/") replaced with "__"
def min_path_wo_slash(file_path):
    """

    @param file_path:
    @return:
    """
    # Only retrieve the file_path after the sample library path
    file_path_minimized = file_path.replace(GlobalConfig.SAMPLE_LIBRARY + "/", "")

    # Replace the slashes with __, otherwise the path where to write is not recognised
    return file_path_minimized.replace("/", "__")
