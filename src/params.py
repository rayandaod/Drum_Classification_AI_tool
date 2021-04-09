import numpy as np
from pathlib import Path

DRUM_TYPES = ["kick", "snare", "hat", "tom"]

# Paths
DATAFRAME_FILENAME = 'dataset.pkl'
here = Path(__file__).parent
DATA_PATH = here / '../data/'

# Preprocessing parameters
DEFAULT_SR = 22050
MAX_SAMPLE_DURATION = 5  # seconds
MAX_FRAMES = 44     # About 1s of audio max, given librosa's default hop_length (= 512 samples): 44*512=22'528 samples
MAX_RMS_CUTOFF = 0.02   # If there is no frame with RMS >= MAX_RMS_CUTOFF within MAX_FRAMES, we'll filter it out

# Feature extraction parameters
DEFAULT_START_ATTACK_THRESHOLD = 0.02
DEFAULT_FRAME_LENGTH = 2048
SUMMARY_OPS = {
    'avg': np.mean,
    'max': np.max,
    'min': np.min,
    'std': np.std,
    'zcr': (lambda arr: len(np.where(np.diff(np.sign(arr)))[0]) / float(len(arr)))
}
