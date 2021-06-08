import warnings
import os
import audioread
import audioop
import librosa
import logging

from config import GlobalConfig

logger = logging.getLogger(__name__)


# Check that the given file path does not already exists.
# If it does, add "_new" at the end (before the extension)
# If it doesn't, simply return the originally given path
def can_write(file_path):
    if os.path.exists(file_path):
        warnings.warn("The given path already exists, adding \"_new\" at the end.")
        path_split = os.path.splitext(file_path)
        return can_write(path_split[0] + "_new" + path_split[1])
    else:
        return file_path


def can_load_audio(path_string):
    if not os.path.isfile(path_string):
        return False
    try:
        librosa.core.load(path_string, mono=True, res_type='kaiser_fast', duration=.01)
    except (audioread.NoBackendError, audioread.DecodeError, EOFError, FileNotFoundError, ValueError, audioop.error):
        logger.warning(f'Skipping {path_string}, unreadable')
        return False
    return True


def load_raw_audio(path_string, sr=GlobalConfig.DEFAULT_SR, offset=0, duration=None, fast=False):
    """
    Mostly pass-through to librosa, but more defensively
    """
    try:
        audio_data, sr = librosa.core.load(path_string, sr=sr, mono=True, offset=offset, duration=duration,
                                           res_type=('kaiser_fast' if fast else 'kaiser_best'))
    except (audioread.NoBackendError, audioread.DecodeError, EOFError, FileNotFoundError, ValueError, audioop.error):
        logger.warning(f'Can\'t read {path_string}')
        return None

    if (duration is None and audio_data.shape[0] > 0) \
            or (duration is not None and audio_data.shape[0] + 1 >= int(sr * duration)):
        return audio_data
    else:
        logger.warning(f'Can\'t load {path_string} due to length, {audio_data.shape[0]} {int(sr * duration)} '
                       f'{duration} {sr}')
        return None
