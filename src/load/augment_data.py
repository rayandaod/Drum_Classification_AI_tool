import librosa
import os
import sys
import soundfile as sf

sys.path.append(os.path.abspath(os.path.join('')))

import audio_tools
import global_helper
from config import *
from paths import *


def time_stretch(raw_audio, stretch_factor, file_path, sr=GlobalConfig.DEFAULT_SR):
    """

    @param raw_audio:
    @param stretch_factor:
    @param file_path:
    @param sr:
    @return:
    """
    # Generate the time-stretched version of the given raw audio
    time_stretched_wav = librosa.effects.time_stretch(raw_audio, stretch_factor)

    # Retrieve a minimised version of a file path, with the slashes ("/") replaced with "__"
    file_path_wo_slash = min_path_wo_slash(file_path)

    # Create the new file path (to point to the data/time_stretched/ folder)
    new_file_path = os.path.join(TIME_STRETCHED_PATH, "{}_{}".format(stretch_factor, file_path_wo_slash))

    # Write the time_stretched audio at the right path
    new_file_path = audio_tools.can_write(new_file_path)
    sf.write(new_file_path, time_stretched_wav, samplerate=sr)

    return new_file_path


def pitch_shift(raw_audio, semitones, file_path, sr=GlobalConfig.DEFAULT_SR):
    """

    @param raw_audio:
    @param semitones:
    @param file_path:
    @param sr:
    @return:
    """
    # Generate the pitch-shifted version of the given raw audio
    pitch_shifted_wav = librosa.effects.pitch_shift(raw_audio, sr, semitones)

    # Retrieve a minimised version of a file path, with the slashes ("/") replaced with "__"
    file_path_wo_slash = min_path_wo_slash(file_path)

    # Create the new file path (to point to the data/time_stretched/ folder)
    new_file_path = os.path.join(PITCH_SHIFTED_PATH, "{}_{}".format(semitones, file_path_wo_slash))

    # Write the time_stretched audio at the right path
    new_file_path = audio_tools.can_write(new_file_path)
    sf.write(new_file_path, pitch_shifted_wav, samplerate=sr)

    return new_file_path


# Retrieve a minimised version of a file path, with the slashes ("/") replaced with "__"
def min_path_wo_slash(file_path):
    """

    @param file_path:
    @return:
    """
    # Only retrieve the file_path after the sample library path
    file_path_minimized = file_path.replace(GlobalConfig.SAMPLE_LIBRARY + "/", "")

    # Replace the slashes with __, otherwise the path where to write is not recognised
    return file_path_minimized.replace("/", "__")


def augment_data(dataset_folder, min_per_class=DataAugmentConfig.MIN_PER_CLASS,
                 pitch_shifting_time_stretching_repartition=DataAugmentConfig.AUGMENTATION_REPARTITION,
                 pitch_shifting_range=DataAugmentConfig.PITCH_SHIFTING_RANGE,
                 time_stretching_range=DataAugmentConfig.TIME_STRETCHING_RANGE):
    """

    @param dataset_folder:
    @param min_per_class:
    @param pitch_shifting_time_stretching_repartition:
    @param pitch_shifting_range:
    @param time_stretching_range:
    @return:
    """

    # Load the drum dataframe
    drums_df = global_helper.load_dataset(dataset_folder, DATASET_FILENAME)

    # Check class imbalance



if __name__ == "__main__":
    dataset_folder = global_helper.parse_args(global_helper.global_parser()).folder
    augment_data(dataset_folder)
