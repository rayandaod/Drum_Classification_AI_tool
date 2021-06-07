import json

import librosa

from config import *


def assign_class(absolute_path, file_stem, blacklist_path, ignore_path):
    """
    Assigns a class to the sample, excluding keywords in blacklist.

    @param absolute_path: The absolute path of the current sample
    @param file_stem: The name of the current sample
    @param ignore_path:
    @param blacklist_path:
    @return The assigned class, among those in DRUM_TYPES
    """
    assignment = {"drum_type": None,
                  "blacklisted": None,
                  "ignored": None}

    for drum_type in GlobalConfig.DRUM_TYPES:
        # That way, we first check the file stem (in case the latter contains "hat" and the absolute path contains
        # "kick" for example)
        if drum_type in file_stem.lower() or drum_type in absolute_path.lower():

            blacklist_file = open(blacklist_path)
            for line in blacklist_file:
                blacklist = line.split(",")
                for b in blacklist:
                    if b in absolute_path.lower():
                        assignment["blacklisted"] = absolute_path
                        return assignment

            ignoring_file = open(ignore_path)
            for line in ignoring_file:
                to_ignore = line.split(",")
                for ig in to_ignore:
                    if ig in absolute_path.lower():
                        absolute_path.replace(ig, "")
                        for dt in GlobalConfig.DRUM_TYPES:
                            if dt in file_stem.lower() or dt in absolute_path.lower():
                                assignment["drum_type"] = dt
                                return assignment
                            else:
                                assignment["ignored"] = absolute_path
                                return assignment

            assignment["drum_type"] = drum_type
            return assignment

    return assignment


def detect_onsets(raw_audio, sr=GlobalConfig.DEFAULT_SR):
    """
    Finds the first onset of the sound, returns a good start time and end time that isolates the sound

    :param raw_audio: np array of audio data, from librosa.load_drums
    :param sr: sample rate
    :return: dict with 'start' and 'end', in seconds
    """
    start = 0.0
    end = None

    # Add an empty second so that the beginning onset is recognized
    silence_to_add = 1.0
    raw_audio = np.append(np.zeros(int(silence_to_add * sr)), raw_audio)

    # Spectral flux
    hop_length = int(sr * PreprocessingConfig.SR_FRACTION_FOR_TRIM)
    onsets = librosa.onset.onset_detect(y=raw_audio, sr=sr, hop_length=hop_length, units='time')

    if len(onsets) == 0:
        return {'start': start, 'end': end}
    elif len(onsets) > 1:
        # If there are multiple onsets, cut it off just before the second one
        end = onsets[1] - (silence_to_add + 0.01)

    start = max(onsets[0] - (silence_to_add + 0.01), 0.0)
    return start, end


def is_too_quiet(raw_audio, max_frames=GlobalConfig.MAX_FRAMES, min_required_rms=GlobalConfig.MIN_REQ_RMS):
    """


    @param raw_audio:
    @param max_frames:
    @param min_required_rms:
    @return: True (too quiet) if the first max_frames frames have an RMS value below the min_required_rms threshold,
    False (not too quiet) otherwise
    """
    frame_length = min(GlobalConfig.DEFAULT_FRAME_LENGTH, len(raw_audio))
    S, _ = librosa.magphase(librosa.stft(y=raw_audio, n_fft=frame_length))
    rms = librosa.feature.rms(S=S, frame_length=frame_length,
                              hop_length=frame_length//GlobalConfig.DEFAULT_HOP_LENGTH_DIV_FACTOR)[0]
    return max(rms[:max_frames]) < min_required_rms


def create_metadata(drums_df, input_dir_path, blacklisted_files, ignored_files, too_long_files, quiet_outliers_list,
                    folder_path, metadata_filename):
    """


    @param drums_df:
    @param input_dir_path:
    @param blacklisted_files:
    @param ignored_files:
    @param too_long_files:
    @param quiet_outliers_list:
    @param folder_path:
    @param metadata_filename:
    @return:
    """
    # Retrieve the current columns of the dataframe
    # Doing the following otherwise drums_df_with_features.columns is not serializable
    columns = []
    for col_name in drums_df.columns:
        columns.append(col_name)

    # Create the metadata.json
    metadata = {
        "sample_library_path": input_dir_path,
        "n_samples": len(drums_df.index),
        "classes": {},
        "n_columns": len(columns),
        "columns": columns,
        "n_blacklisted_files": len(blacklisted_files),
        "blacklisted_files": blacklisted_files,
        "n_ignored_files": len(ignored_files),
        "ignored_files": ignored_files,
        "n_too_long_files": len(too_long_files),
        "too_long_files": too_long_files,
        "n_quiet_outliers": len(quiet_outliers_list),
        "quiet_outliers": quiet_outliers_list,
        "ok": drums_df['audio_path'].tolist()
    }

    # Add the class balance
    for drum_type in GlobalConfig.DRUM_TYPES:
        metadata["classes"][drum_type] = len(drums_df[drums_df["drum_type"] == drum_type].index)

    # Create the json file
    with open(folder_path / metadata_filename, 'w') as outfile:
        json.dump(metadata, outfile)
