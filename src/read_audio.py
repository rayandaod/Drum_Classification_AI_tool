import os
import audioread
import audioop
import librosa
import logging
import numpy as np

import params

logger = logging.getLogger(__name__)


def can_load_audio(path_string):
    if not os.path.isfile(path_string):
        return False
    try:
        librosa.core.load(path_string, mono=True, res_type='kaiser_fast', duration=.01)
    except (audioread.NoBackendError, audioread.DecodeError, EOFError, FileNotFoundError, ValueError, audioop.error):
        logger.warning(f'Skipping {path_string}, unreadable')
        return False
    return True


def load_raw_audio(path_string, sr=params.DEFAULT_SR, offset=0, duration=None, fast=False):
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


def load_clip_audio(clip, sr=params.DEFAULT_SR):
    """
    Clip is a row of a dataframe with a start_time, end_time, audio_path, and new_duration
    """
    duration = None if clip.end_time is None or np.isnan(clip.end_time) else (clip.end_time - clip.start_time)
    return load_raw_audio(clip.audio_path, sr=sr, offset=clip.start_time, duration=duration)
