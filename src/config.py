import numpy as np
from pathlib import Path


class GlobalConfig:
    DRUM_TYPES = ['kick', 'snare', 'hat', 'tom']
    DEFAULT_SR = 22050


class PathConfig:
    SAMPLE_LIBRARY = "/Users/rayandaod/Documents/Prod/My_samples"
    here = Path(__file__).parent
    DATA_PATH = here / '../data/'

    DATAFRAME_FILENAME = 'dataset.pkl'
    PICKLE_DATASET_PATH = DATA_PATH.joinpath(DATAFRAME_FILENAME)

    DATAFRAME_NOT_CAPED_FILENAME = 'dataset_not_caped.pkl'
    PICKLE_DATAFRAME_NOT_CAPED_PATH = DATA_PATH.joinpath(DATAFRAME_NOT_CAPED_FILENAME)

    QUIET_OUTLIERS_PATH = DATA_PATH / "quiet_outliers.txt"
    BLACKLISTED_FILES_PATH = DATA_PATH / "blacklisted_files.txt"
    IGNORE_PATH = DATA_PATH / "ignore.txt"
    BLACKLIST_PATH = DATA_PATH / "blacklist.txt"

    DATAFRAME_WITH_FEATURES_FILENAME = 'dataset_features.pkl'
    PICKLE_DATASET_WITH_FEATURES_PATH = DATA_PATH.joinpath(DATAFRAME_WITH_FEATURES_FILENAME)

    IMPUTATER_PATH = DATA_PATH / 'imputer.pkl'
    SCALER_PATH = DATA_PATH / 'scaler.pkl'


class PreprocessingConfig:
    SR_FRACTION_FOR_TRIM = 1 / 200.0
    MAX_SAMPLE_DURATION = 5  # seconds
    MAX_FRAMES = 44  # About 1s of audio max, given librosa's default hop_length (= 512 samples): 44*512=22'528 samples
    MAX_RMS_CUTOFF = 0.02  # If there is no frame with RMS >= MAX_RMS_CUTOFF within MAX_FRAMES, we'll filter it out


class FeatureConfig:
    DEFAULT_START_ATTACK_THRESHOLD = 0.02
    DEFAULT_FRAME_LENGTH = 2048
    MAX_FRAME = PreprocessingConfig.MAX_FRAMES
    MAX_RMS_CUTOFF = PreprocessingConfig.MAX_RMS_CUTOFF
    SUMMARY_OPS = {
        'avg': np.mean,
        'max': np.max,
        'min': np.min,
        'std': np.std,
        'zcr': (lambda arr: len(np.where(np.diff(np.sign(arr)))[0]) / float(len(arr)))
    }


class TrainingConfig:
    N_SAMPLES_PER_CLASS = 750
