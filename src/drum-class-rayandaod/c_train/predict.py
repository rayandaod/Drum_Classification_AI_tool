import sys
import os
import torch
from torch import Tensor

sys.path.append(os.path.abspath(os.path.join('')))

from a_data import load
from b_features import extract
from z_helpers.paths import *
from config import *


def predict(path, dataset_folder_name, model_folder_name):
    # Load the given audio files
    drums_df, unreadable_files, _, _, _, quiet_outliers = load.load(path, eval=True)
    print(f'Unreadable: {unreadable_files}')
    print(f'Too quiet: {quiet_outliers}')

    # Extract their features
    drums_df_with_features, _ = extract.load_extract_from(None, drums_df)

    # Load the saved model
    nn = torch.load(MODELS / dataset_folder_name / model_folder_name / MODEL_FILENAME)
    nn.eval()
    if nn.name != "CNN":
        drums_df_with_features = drums_df_with_features.drop(columns=['melS'])

    # Predict their classes
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
    drum_types = predict(path='/Users/rayandaod/Documents/Prod/My_samples/Medasin Overdose 5/mskrb_drums/MSKRB_hihats',
                         dataset_folder_name='20210609-025547-My_samples',
                         model_folder_name='NN_20210609-155727')
    print(drum_types)
