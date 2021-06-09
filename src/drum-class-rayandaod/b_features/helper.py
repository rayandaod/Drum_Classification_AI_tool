from config import *
from z_helpers import audio_tools


def only_req_rms_frames(rms, audio_path, max_frames=GlobalConfig.MAX_FRAMES, min_req_rms=GlobalConfig.MIN_REQ_RMS):
    # Keep the RMS values only for the first max_frames frames
    rms = rms[:max_frames]

    # If the signal is too quiet, spectral/mfcc features might not be accurate in places (also, 0.0 will
    # yield nans after log) Discard quiet frames from here on out
    valid_frames = rms >= min_req_rms

    # Reminder: valid_frames is a boolean array
    # Check that the number of valid frames is greater than 0, otherwise the sample should have been removed previously
    assert sum(valid_frames) > 0, f'Too quiet for analysis: {audio_path}'

    return rms[valid_frames], valid_frames
