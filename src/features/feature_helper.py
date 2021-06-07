from config import *


def trim_rms(rms, max_frames=FeatureConfig.MAX_FRAME, max_rms_cutoff=FeatureConfig.MAX_RMS_CUTOFF):
    rms = rms[:max_frames]

    # If the signal is too quiet, spectral/mfcc features might not be accurate in places (also, 0.0 will
    # yield nans after log) Discard quiet frames from here on out
    valid_frames = rms >= max_rms_cutoff
    rms = rms[valid_frames]

    # Reminder: valid_frames is a boolean array
    # Check that the number of valid frames is greater than 0, otherwise the sample should have been removed previously
    assert sum(valid_frames) > 0, 'sound too quiet for analysis, filter out using filter_quiet_outliers()'

    return rms, valid_frames
