import sys
import os
import torch
import pickle
from torch import Tensor

sys.path.append(os.path.abspath(os.path.join('')))

from a_data import load
from b_features import extract
from z_helpers.paths import *
from z_helpers import global_helper
from config import *


def predict(some_path, dataset_folder, model_folder_name, is_sample_dict=False):
    # Load the given audio files
    drums_df, unreadable_files, _, _, _, quiet_outliers = load.load(some_path, eval=True, is_sample_dict=is_sample_dict)

    if is_sample_dict:
        dict_indices = drums_df['dict_index'].tolist()
        drums_df = global_helper.drop_columns(drums_df, columns=['dict_index'])

    print(f'Unreadable: {unreadable_files}')
    print(f'Too quiet: {quiet_outliers}')

    # Extract their features
    drums_df_with_features, _ = extract.load_extract_from(None, drums_df)

    model_folder_path = MODELS_PATH / dataset_folder / model_folder_name
    predictions = []

    # If the model folder starts with NN or CNN, use torch and stuff
    model_prefix = model_folder_name.split('_')[0]
    if model_prefix == 'NN' or model_prefix == 'CNN':
        if model_prefix == "CNN":
            nn = torch.load(model_folder_path / MODEL_FILENAME)
            nn.eval()
            drums_df_with_features = drums_df_with_features['melS']
        else:
            drums_df_with_features = drums_df_with_features.drop(columns=['melS'])
            nn = torch.load(model_folder_path / MODEL_FILENAME)
            nn.eval()
        # Predict their classes
        with torch.no_grad():
            # Generate prediction
            for row in drums_df_with_features.values.tolist():
                prediction = nn(Tensor([row]))

                # Predicted class value using argmax
                predicted_class = ['hat', 'tom', 'snare', 'kick'][np.argmax(prediction)]
                predictions.append(predicted_class)
    # Else, use other stuff
    else:
        with open(model_folder_path / MODEL_FILENAME, 'rb') as file:
            pickle_model = pickle.load(file)

        # Remove the melS feature since we are not predicting with the CNN model here
        drums_df_with_features = global_helper.drop_columns(drums_df_with_features, columns=['melS'])
        drums_np = drums_df_with_features.to_numpy()

        # Load imputer and scaler and... impute and scale lol
        imp = pickle.load(open(DATA / DATASETS_PATH / dataset_folder / IMPUTATER_FILENAME, 'rb'))
        scaler = pickle.load(open(DATA / DATASETS_PATH / dataset_folder / SCALER_FILENAME, 'rb'))
        drums_np = imp.transform(drums_np)
        drums_np = scaler.transform(drums_np)

        # Predict
        prediction = pickle_model.predict(drums_np)
        predictions = [['hat', 'tom', 'snare', 'kick'][pred] for pred in prediction]

    if is_sample_dict:
        predictions = dict(zip(dict_indices, predictions))

    return predictions, unreadable_files, quiet_outliers


if __name__ == "__main__":
    drum_types = predict(
        some_path='/Users/rayandaod/Documents/Prod/My_samples/AP11 Sample Pack/Snares',
        dataset_folder='20210609-025547-My_samples',
        model_folder_name='RF_20210609-234422')
    print(drum_types)
