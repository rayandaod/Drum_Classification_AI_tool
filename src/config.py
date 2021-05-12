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
    RELOAD = False
    VERBOSE = False


class PathConfig:
    SAMPLE_LIBRARY = "/Users/rayandaod/Documents/Prod/My_samples/"
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


class PreprocessingConfig:
    SR_FRACTION_FOR_TRIM = 1 / 200.0
    MAX_SAMPLE_DURATION = 5  # seconds
    MAX_FRAMES = 44  # About 1s of audio max, given librosa's default hop_length (= 512 samples): 44*512=22'528 samples
    MAX_RMS_CUTOFF = 0.02  # If there is no frame with RMS >= MAX_RMS_CUTOFF within MAX_FRAMES, we'll filter it out


# class DataAugmentationConfig:


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


class DataPrepConfig:
    def __init__(self):
        # Data preparation variables
        self.N_SAMPLES_PER_CLASS = None
        self.VALIDATION_SET_RATIO = 0.1


class TrainingConfig:
    class SimpleTrainingConfig:
        MODELS = {
            'lr': LinearRegression(),
            'svc': SVC(),
            'random_forest': RandomForestClassifier(n_estimators=500),
            'gb': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3,
                                             random_state=GlobalConfig.RANDOM_STATE),
            'knn': KNeighborsClassifier(),
        }

        iterative_imputer = IterativeImputer(max_iter=25, random_state=GlobalConfig.RANDOM_STATE)

        grid_searches = {
            'sweep_rf': {
                'model_type': 'random forest',
                "n_est_values": [100, 200, 500, 1000],
                "depth_values": [10]
            }
        }

    # Implemented this way in order to easily serialize it to JSON
    class NNTrainingConfig:
        def __init__(self):
            self.EPOCHS = 300  # Plot the learning curves
            self.BATCH_SIZE = 16
            self.LEARNING_RATE = 0.0007  # test factor 10 below and above, learning rate schedule?
            self.DROPOUT_P = 0.2
