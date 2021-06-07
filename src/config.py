import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from pathlib import Path

import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CNN_INPUT_SIZE = (128, 256)


class GlobalConfig:
    SAMPLE_LIBRARY = "/Users/rayandaod/Documents/Prod/My_samples/KSHMR Vol. 3"
    DRUM_TYPES = ['kick', 'snare', 'hat', 'tom']

    DEFAULT_SR = 22050

    DEFAULT_FRAME_LENGTH = 2048
    DEFAULT_HOP_LENGTH_DIV_FACTOR = 4
    MAX_FRAMES = 44
    MAX_RMS_CUTOFF = 0.02

    RANDOM_STATE = None
    RELOAD = False
    VERBOSE = False


class PreprocessingConfig:
    SR_FRACTION_FOR_TRIM = 1 / 200.0
    MAX_SAMPLE_DURATION = 5  # seconds


# class DataAugmentationConfig:


class FeatureConfig:
    DEFAULT_START_ATTACK_THRESHOLD = 0.02
    SUMMARY_OPS = {
        'avg': np.mean,
        'max': np.max,
        'min': np.min,
        'std': np.std,
        'zcr': (lambda arr: len(np.where(np.diff(np.sign(arr)))[0]) / float(len(arr)))
    }


class DataAugmentConfig:
    MIN_PER_CLASS = 800
    AUGMENTATION_REPARTITION = 0.5
    PITCH_SHIFTING_RANGE = np.arange(-5, 5)
    TIME_STRETCHING_RANGE = np.arange(0.4, 1.5, 0.1)


class DataPrepConfig:
    def __init__(self):
        # Data preparation variables
        self.N_SAMPLES_PER_CLASS = None
        self.VALIDATION_SET_RATIO = 0.1
        self.CNN_BATCH_SIZE = 64
        self.CNN_VAL_BATCH_SIZE = 64


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
            self.N_INPUT = None
            self.EPOCHS = 300  # Plot the learning curves
            self.BATCH_SIZE = 16
            self.LEARNING_RATE = 0.0007  # test factor 10 below and above, learning rate schedule?
            self.DROPOUT_P = 0.2

    # Implemented this way in order to easily serialize it to JSON
    class CNNTrainingConfig:
        def __init__(self):
            self.MAX_EPOCHS = 300
            self.EARLY_STOPPING = 10
            self.LEARNING_RATE = 0.002
            self.MOMENTUM = 0.6
            self.LOG_INTERVAL = 20
