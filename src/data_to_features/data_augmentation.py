import librosa
import os
import soundfile as sf

from src import *
from config import GlobalConfig
from data_to_features import helper


def time_stretch(raw_audio, stretch_factor, file_path, sr=GlobalConfig.DEFAULT_SR):

    # Generate the time-stretched version of the given raw audio
    time_stretched_wav = librosa.effects.time_stretch(raw_audio, stretch_factor)

    # Retrieve a minimised version of a file path, with the slashes ("/") replaced with "__"
    file_path_wo_slash = min_path_wo_slash(file_path)

    # Create the new file path (to point to the data/time_stretched/ folder)
    new_file_path = os.path.join(TIME_STRETCHED_PATH, "{}_{}".format(stretch_factor, file_path_wo_slash))

    # Write the time_stretched audio at the right path
    new_file_path = helper.can_write(new_file_path)
    sf.write(new_file_path, time_stretched_wav, samplerate=sr)

    return new_file_path


def pitch_shift(raw_audio, semitones, file_path, sr=GlobalConfig.DEFAULT_SR):
    # Generate the pitch-shifted version of the given raw audio
    pitch_shifted_wav = librosa.effects.pitch_shift(raw_audio, sr, semitones)

    # Retrieve a minimised version of a file path, with the slashes ("/") replaced with "__"
    file_path_wo_slash = min_path_wo_slash(file_path)

    # Create the new file path (to point to the data/time_stretched/ folder)
    new_file_path = os.path.join(PITCH_SHIFTED_PATH, "{}_{}".format(semitones, file_path_wo_slash))

    # Write the time_stretched audio at the right path
    new_file_path = helper.can_write(new_file_path)
    sf.write(new_file_path, pitch_shifted_wav, samplerate=sr)

    return new_file_path


# Retrieve a minimised version of a file path, with the slashes ("/") replaced with "__"
def min_path_wo_slash(file_path):
    # Only retrieve the file_path after the sample library path
    file_path_minimized = file_path.replace(SAMPLE_LIBRARY + "/", "")

    # Replace the slashes with __, otherwise the path where to write is not recognised
    return file_path_minimized.replace("/", "__")


def augment_data(drums_df, dataset_folder):
    # Check class imbalance

    return drums_df
