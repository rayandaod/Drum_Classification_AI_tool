import logging

from pathlib import Path

from config import GlobalConfig
from argparse import ArgumentParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


here = Path(__file__).parent
DATA_PATH = here / '../data/'

DATASET_FILENAME = 'dataset.pkl'
PICKLE_DATASETS_PATH = DATA_PATH / 'datasets/'

# DATAFRAME_NOT_CAPED_FILENAME = 'dataset_not_caped.pkl'
# PICKLE_DATAFRAME_NOT_CAPED_PATH = DATA_PATH.joinpath(DATAFRAME_NOT_CAPED_FILENAME)

QUIET_OUTLIERS_FILENAME = "quiet_outliers.txt"
BLACKLISTED_FILES_FILENAME = "blacklisted_files.txt"
IGNORE_PATH = DATA_PATH / "to_ignore.txt"
BLACKLIST_PATH = DATA_PATH / "to_blacklist.txt"

METADATA_JSON_FILENAME = "metadata.json"

AUGMENTED_DATA_PATH = DATA_PATH / "augmented_data/"
TIME_STRETCHED_PATH = AUGMENTED_DATA_PATH / "time_stretched/"
PITCH_SHIFTED_PATH = AUGMENTED_DATA_PATH / "pitch_shifted/"

DATASET_WITH_FEATURES_FILENAME = 'dataset_features.pkl'

IMPUTATER_FILENAME = 'imputer.pkl'
SCALER_FILENAME = 'scaler.pkl'

MODELS_PATH = here / '../models/'
MODEL_FILENAME = 'model.pth'


def global_parser():
    parser = ArgumentParser()
    parser.add_argument('--reload', action='store_true', dest='reload',
                        help='Reload the drum library and extract the features again.')
    parser.add_argument('--old', type=str, default=None, help='Select an already loaded dataset')
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
