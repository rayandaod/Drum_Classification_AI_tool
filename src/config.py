import numpy as np
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
            self.N_INPUT = None
            self.EPOCHS = 300  # Plot the learning curves
            self.BATCH_SIZE = 16
            self.LEARNING_RATE = 0.0007  # test factor 10 below and above, learning rate schedule?
            self.DROPOUT_P = 0.2
