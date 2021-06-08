import librosa
from typing import Dict

from config import *
import b_features.feature_helper as feature_helper


def extract_features_from_single(raw_audio, audio_path):
    assert raw_audio.size != 0
    features: Dict[str, float] = dict()

    # Handle audio samples shorter than one frame
    frame_length = min(GlobalConfig.DEFAULT_FRAME_LENGTH, len(raw_audio))

    # Retrieve the magnitude of the sample's STFT
    S, _ = librosa.magphase(librosa.stft(y=raw_audio, n_fft=frame_length))

    # Compute root-mean-square (RMS) value for each frame, taking the frame_length computed before
    # (2048 samples or less), and the hop_length being frame_length//DEFAULT_HOP_LENGTH_DIV_FACTOR
    rms = librosa.feature.rms(S=S, frame_length=frame_length,
                              hop_length=frame_length//GlobalConfig.DEFAULT_HOP_LENGTH_DIV_FACTOR)[0]

    # MPEG-7 standard features
    features = {**features, **mpeg7_features(rms, frame_length)}

    # For the remainder of features, only focus on frames within 1 second
    rms, valid_frames = feature_helper.only_req_rms_frames(rms, audio_path)
    is_long_enough_for_gradient = len(rms) > 1
    loudest_valid_frame_index = np.argmax(rms)

    # RMS features
    features = {**features, **rms_features(rms, is_long_enough_for_gradient)}

    # Zero-Crossing-Rate feature
    features = {**features, **zero_crossing_rate_features(raw_audio, frame_length, valid_frames,
                                                          loudest_valid_frame_index)}

    # Spectral features
    features = {**features,
                **spectral_features(S, frame_length, valid_frames, is_long_enough_for_gradient,
                                    loudest_valid_frame_index)}

    # MFCC features
    features = {
        **features,
        **mfcc_features(S, valid_frames, is_long_enough_for_gradient, loudest_valid_frame_index)
    }

    # Mel-Spectrogram features
    features = {
        **features, **melspectrogram(S)
    }

    return features


def mpeg7_features(rms, frame_length, SA_threshold=FeatureConfig.DEFAULT_START_ATTACK_THRESHOLD,
                   sr=GlobalConfig.DEFAULT_SR):
    features: Dict[str, float] = dict()

    # Get the index of the maximum amplitude sample
    peak = np.argmax(rms)

    # Get the length of a hop in seconds
    hop_time_length = (frame_length // 4) / sr

    # Compute an array of True, False values, where
    # - True means that the corresponding sample in rms has a value above SA_threshold * the maximum
    # - False means otherwise
    loud_enough = rms >= SA_threshold * rms[peak]

    # Get the corresponding indices in rms
    loud_enough_idx = np.where(loud_enough)[0]
    first_loud_enough = loud_enough_idx[0]
    last_loud_enough = loud_enough_idx[-1]

    # Take the rms array to power 2, element-wise, and get the loud enough-elements associated to it
    power = rms ** 2
    loud_enough_power = power[first_loud_enough: last_loud_enough + 1]

    # Log attack time
    attack_time = peak * hop_time_length - first_loud_enough * hop_time_length
    # If the attack is 0, we can't take the log so pretend the attack is half of one frame
    log_attack_t = np.log10(attack_time) if attack_time > 0 else np.log10(hop_time_length / 2.0)
    features["log_attack_time"] = log_attack_t

    # - Temporal centroid
    temp_cent = np.sum(loud_enough_power * np.linspace(0.0, 1.0, len(loud_enough_power))) / np.sum(loud_enough_power) \
        if np.sum(loud_enough_power) > 0 else np.NaN
    features["temporal_centroid"] = temp_cent

    # - Log attack time - Temporal centroid ratio
    features['lat_tc_ratio'] = log_attack_t / temp_cent if temp_cent > 0 else np.NaN

    # TODO: check
    # - Loud enough duration
    features['duration'] = hop_time_length * len(loud_enough_power)

    # TODO: check
    # - Release
    features['release'] = hop_time_length * (last_loud_enough - peak)

    return features


def rms_features(rms, is_long_enough_for_gradient):
    features: Dict[str, float] = dict()

    # Peak-to-average ratio
    features['crest_factor'] = rms.max() / rms.mean()

    # Log RMS
    log_rms = np.log10(rms)
    for op in ['avg', 'std', 'min', 'max']:
        features[f'log_rms_{op}'] = FeatureConfig.SUMMARY_OPS[op](log_rms)

    # Log RMS gradient (but only for drum sounds with more than 1 frame, otherwise put NaN and handle later in training)
    if is_long_enough_for_gradient:
        log_rms_d = np.gradient(log_rms)
        for op in FeatureConfig.SUMMARY_OPS.keys():
            features[f'log_rms_d_{op}'] = FeatureConfig.SUMMARY_OPS[op](log_rms_d)
    else:
        for op in FeatureConfig.SUMMARY_OPS.keys():
            features[f'log_rms_d_{op}'] = np.NaN

    return features


def zero_crossing_rate_features(raw_audio, frame_length, valid_frames, loudest_valid_frame_index):
    features: Dict[str, float] = dict()
    zcr = librosa.feature.zero_crossing_rate(raw_audio, frame_length=frame_length, hop_length=frame_length // 4)
    zcr = zcr[0][:GlobalConfig.MAX_FRAMES][valid_frames]
    for op in ['avg', 'min', 'max', 'std']:
        features[f'zcr_{op}'] = FeatureConfig.SUMMARY_OPS[op](zcr)
    features['zcr_loudest'] = zcr[loudest_valid_frame_index]
    return features


# TODO: add spectral contrast ?
def spectral_features(S, frame_length, valid_frames, is_long_enough_for_gradient, loudest_valid_frame_index):
    features: Dict[str, float] = dict()

    log_spec_cent = np.log10(librosa.feature.spectral_centroid(
        S=S, n_fft=frame_length, hop_length=frame_length // 4)[0][:GlobalConfig.MAX_FRAMES][valid_frames])
    for op in ['avg', 'min', 'max', 'std']:
        features[f'log_spec_cent_{op}'] = FeatureConfig.SUMMARY_OPS[op](log_spec_cent)
    features['log_spec_cent_loudest'] = log_spec_cent[loudest_valid_frame_index]

    if is_long_enough_for_gradient:
        log_spec_cent_d = np.gradient(log_spec_cent)
    for op in FeatureConfig.SUMMARY_OPS.keys():
        features[f'log_spec_cent_d_{op}'] = FeatureConfig.SUMMARY_OPS[op](
            log_spec_cent_d) if is_long_enough_for_gradient \
            else np.NaN

    log_spec_band = np.log10(librosa.feature.spectral_bandwidth(S=S, n_fft=frame_length, hop_length=frame_length // 4)
                             [0][:GlobalConfig.MAX_FRAMES][valid_frames])
    for op in ['avg', 'min', 'max', 'std']:
        features[f'log_spec_band_{op}'] = FeatureConfig.SUMMARY_OPS[op](log_spec_band)
    features['log_spec_band_d_avg'] = np.mean(np.gradient(log_spec_band)) if is_long_enough_for_gradient else np.NaN

    spec_flat = librosa.feature.spectral_flatness(
        S=S, n_fft=frame_length, hop_length=frame_length // 4)[0][:GlobalConfig.MAX_FRAMES][valid_frames]
    for op in ['avg', 'min', 'max', 'std']:
        features[f'spec_flat_{op}'] = FeatureConfig.SUMMARY_OPS[op](spec_flat)
    features['spec_flat_loudest'] = spec_flat[loudest_valid_frame_index]
    features['spec_flat_d_avg'] = np.mean(np.gradient(spec_flat)) if is_long_enough_for_gradient else np.NaN

    for roll_percent in [.15, .85]:
        spec_rolloff = librosa.feature.spectral_rolloff(
            S=S, roll_percent=roll_percent, n_fft=frame_length, hop_length=frame_length // 4
        )[0][:GlobalConfig.MAX_FRAMES][valid_frames]
        roll_percent_int = int(100 * roll_percent)
        features[f'log_spec_rolloff_{roll_percent_int}_loudest'] = np.log10(spec_rolloff[loudest_valid_frame_index]) \
            if spec_rolloff[loudest_valid_frame_index] > 0.0 else np.NaN
        # For some reason some sounds give random 0.0s for the spectral rolloff of certain frames.
        # After log these are -inf and need to be filtered before taking the min
        log_spec_rolloff = np.log10(spec_rolloff[spec_rolloff != 0.0])
        features[f'log_spec_rolloff_{roll_percent_int}_max'] = np.max(log_spec_rolloff) \
            if len(log_spec_rolloff) > 0 else np.NaN
        features[f'log_spec_rolloff_{roll_percent_int}_min'] = np.min(log_spec_rolloff) \
            if len(log_spec_rolloff) > 0 else np.NaN

    return features


# TODO: add zcr?
def mfcc_features(S, valid_frames, is_long_enough_for_gradient, loudest_valid_frame, n_mfcc=13):
    features: Dict[str, float] = dict()

    # Trim the first mfcc because it's basically volume
    mfccs = librosa.feature.mfcc(S=S, n_mfcc=n_mfcc)[1:, :GlobalConfig.MAX_FRAMES][:, valid_frames]
    n_mfcc -= 1

    # Compute once because it's faster
    transformed_mfcc = {
        'avg': FeatureConfig.SUMMARY_OPS['avg'](mfccs, axis=1),
        'min': FeatureConfig.SUMMARY_OPS['min'](mfccs, axis=1),
        'max': FeatureConfig.SUMMARY_OPS['max'](mfccs, axis=1),
        'std': FeatureConfig.SUMMARY_OPS['std'](mfccs, axis=1),
        'loudest': mfccs[:, loudest_valid_frame]
    }

    if is_long_enough_for_gradient:
        mfcc_d_avg = np.mean(np.gradient(mfccs, axis=1), axis=1)

    for n in range(n_mfcc):
        for op in ['avg', 'min', 'max', 'std']:
            features[f'mfcc_{n}_{op}'] = transformed_mfcc[op][n]
        features[f'mfcc_{n}_loudest'] = transformed_mfcc['loudest'][n]
        features[f'mfcc_{n}_d_avg'] = mfcc_d_avg[n] if is_long_enough_for_gradient else np.NaN

    return features


def melspectrogram(S):
    features: Dict[str, float] = dict()
    mel_S = librosa.feature.melspectrogram(S=S)
    features["melS"] = mel_S
    return features
