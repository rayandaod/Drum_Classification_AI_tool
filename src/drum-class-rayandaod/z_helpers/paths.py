from pathlib import Path

here = Path(__file__).parent
ROOT = here / '../../..'
DATA = ROOT / 'data/'
MODELS = ROOT / 'models/'

DATASET_FILENAME = 'dataset.pkl'
PICKLE_DATASETS_PATH = DATA / 'datasets/'

# DATAFRAME_NOT_CAPED_FILENAME = 'dataset_not_caped.pkl'
# PICKLE_DATAFRAME_NOT_CAPED_PATH = DATA_PATH.joinpath(DATAFRAME_NOT_CAPED_FILENAME)

QUIET_OUTLIERS_FILENAME = "quiet_outliers.txt"
BLACKLISTED_FILES_FILENAME = "blacklisted_files.txt"
IGNORE_PATH = DATA / "to_ignore.txt"
BLACKLIST_PATH = DATA / "to_blacklist.txt"

METADATA_JSON_FILENAME = "metadata.json"

AUGMENTED_DATA_PATH = DATA / "augmented_data/"
TIME_STRETCHED_PATH = AUGMENTED_DATA_PATH / "time_stretched/"
PITCH_SHIFTED_PATH = AUGMENTED_DATA_PATH / "pitch_shifted/"

DATASET_WITH_FEATURES_FILENAME = 'dataset_features.pkl'

IMPUTATER_FILENAME = 'imputer.pkl'
SCALER_FILENAME = 'scaler.pkl'

MODEL_FILENAME = 'model.pth'
LOGS_FILENAME = 'logs.txt'
