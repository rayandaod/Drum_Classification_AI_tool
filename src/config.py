import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class GlobalConfig:
    DRUM_TYPES = ['kick', 'snare', 'hat', 'tom']
    DEFAULT_SR = 22050
    RANDOM_STATE = None


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

    AUGMENTED_DATA_PATH = DATA_PATH / "augmented_data/"
    TIME_STRETCHED_PATH = AUGMENTED_DATA_PATH / "time_stretched/"
    PITCH_SHIFTED_PATH = AUGMENTED_DATA_PATH / "pitch_shifted/"

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
    MODELS = {
        'lr': LinearRegression(),
        'svc': SVC(),
        'random_forest': RandomForestClassifier(n_estimators=500),
        'gb': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=0),
        'knn': KNeighborsClassifier()
    }

    iterative_imputer = IterativeImputer(max_iter=25, random_state=GlobalConfig.RANDOM_STATE)

    grid_searches = {
        'sweep_rf': {
            'model_type': 'random forest',
            "n_est_values": [100, 200, 500, 1000],
            "depth_values": [10]
        }
    }
