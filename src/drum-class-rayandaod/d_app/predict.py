import sys
import os
import torch
from torch import Tensor

sys.path.append(os.path.abspath(os.path.join('')))

from a_data import load
from b_features import extract
from z_helpers.paths import *
from config import *


def predict(folder_path, dataset_folder_name, model_folder_name):
    drums_df, blacklisted_files, ignored_files, too_long_files, quiet_outliers = load.load(folder_path)

    if len(blacklisted_files) > 0 or len(ignored_files) > 0 or len(too_long_files) > 0 or len(quiet_outliers) > 0:
        print('File was discarded:')
        print(f'Blacklisted files: {blacklisted_files}')
        print(f'Ignored files: {ignored_files}')
        print(f'Too long files: {too_long_files}')
        print(f'Quiet files: {quiet_outliers}')

    drums_df_with_features, _ = extract.load_extract_from(None, drums_df)
    drums_df_with_features = drums_df_with_features.drop(columns=['drum_type'])
    drums_df_with_features = drums_df_with_features.drop(columns=['melS'])

    nn = torch.load(MODELS / dataset_folder_name / model_folder_name / MODEL_FILENAME)
    nn.eval()

    with torch.no_grad():
        # Generate prediction
        predictions = []
        for row in drums_df_with_features.values.tolist():
            prediction = nn(Tensor([row]))

            # Predicted class value using argmax
            predicted_class = ['hat', 'tom', 'snare', 'kick'][np.argmax(prediction)]
            predictions.append(predicted_class)
    return predictions


if __name__ == "__main__":
    drum_types = predict(folder_path='/Users/rayandaod/Documents/Prod/My_samples/AP11 Sample Pack/Hi-Hats',
                         dataset_folder_name='20210609-025547-My_samples',
                         model_folder_name='NN_20210609-110954')
    print(drum_types)
