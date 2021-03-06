import os
import sys
import time
import json
import pickle

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

sys.path.append(os.path.abspath(os.path.join('')))

from z_helpers import global_helper
from z_helpers.paths import *
from config import *
from c_train import data_prep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS = {
    'lr': LinearRegression(),
    'svc': SVC(),
    'rf': RandomForestClassifier(n_estimators=TrainingConfig.Basic.RF_N_ESTIMATORS),
    'gb': GradientBoostingClassifier(n_estimators=TrainingConfig.Basic.GB_N_ESTIMATORS,
                                     learning_rate=TrainingConfig.Basic.GB_LR,
                                     max_depth=TrainingConfig.Basic.GB_MAX_DEPTH,
                                     random_state=GlobalConfig.RANDOM_STATE),
    'knn': KNeighborsClassifier(),
}


def run(drums_df, dataset_folder, model_key):
    # Prepare the data for training process
    train_X, train_y, test_X, test_y, drum_class_labels, data_prep_config = prepare_data(drums_df, dataset_folder)

    # Train and test
    model, test_X, test_y, drum_class_labels, logs = \
        train_test(MODELS[model_key], train_X, train_y, test_X, test_y, drum_class_labels)

    # Save the model, metadata, and logs
    save_all(model, data_prep_config, dataset_folder, logs)


def prepare_data(drums_df, dataset_folder):
    train_X, train_y, test_X, test_y, drum_class_labels, data_prep_config = data_prep.prep_data_b4_training(drums_df, dataset_folder)
    train_X, test_X = data_prep.impute_and_scale(train_X, test_X, dataset_folder)
    return train_X, train_y, test_X, test_y, drum_class_labels, data_prep_config


def train_test(model, train_X, train_y, test_X, test_y, drum_class_labels):
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    classification_report_string = 'Classification Report:\n' + classification_report(test_y, pred,
                                                                                      target_names=drum_class_labels,
                                                                                      zero_division=0)
    logger.info(classification_report_string)
    return model, test_X, test_y, drum_class_labels, classification_report_string


def save_all(model, data_prep_config, dataset_folder, logs):
    logger.info("Saving model and metadata...")

    # Define the model folder
    model_folder = 'RF_' + time.strftime("%Y%m%d_%H%M%S")
    dataset_in_models = MODELS_PATH / dataset_folder
    model_in_dataset_in_models = dataset_in_models / model_folder
    Path(dataset_in_models).mkdir(parents=True, exist_ok=True)
    Path(model_in_dataset_in_models).mkdir()

    # Save the model
    with open(model_in_dataset_in_models / MODEL_FILENAME, 'wb') as file:
        pickle.dump(model, file)

    # Save the metadata
    metadata_dict = {"MODEL_NAME": "Random Forest"}
    metadata = data_prep_config.__dict__
    metadata_dict.update(metadata)
    with open(model_in_dataset_in_models / METADATA_JSON_FILENAME, 'w') as outfile:
        json.dump(metadata_dict, outfile)

    # Save the logs
    log_file = open(model_in_dataset_in_models / LOGS_FILENAME, "a+")
    log_file.write(logs)
    log_file.close()


# TODO: include grid_search
# def train(drums_df, model_key, dataset_folder, grid_search_key=None):
#     data_prep_config = DataPrepConfig(os.path.basename(os.path.normpath(dataset_folder)))
#     train_X, train_y, test_X, test_y, drum_class_labels = helper.prep_data_b4_training(data_prep_config,
#                                                                                        drums_df,
#                                                                                        dataset_folder)
#     logger.info(f"{model_key}:")
#
#     # grid_search_key is already checked to be part of the grid_search_dict (in TrainingConfig) when launching main.py
#     # it is therefore either None, or an existing key at this point
#     # TODO yes but should be safe in standalone mode too
#     if grid_search_key is not None:
#         grid_search_dict = TrainingConfig.Basic.grid_searches
#         # TODO: only for rf here, must be more general for other classification models
#         for n in grid_search_dict[grid_search_key]["n_est_values"]:
#             logger.info(f"{n} estimators:")
#             model = RandomForestClassifier(n_estimators=n)
#             fit_and_predict(model, train_X, train_y, test_X, test_y, drum_class_labels)
#     else:
#         model = TrainingConfig.Basic.MODELS[model_key]
#         return fit_and_predict(model, train_X, train_y, test_X, test_y, drum_class_labels)


if __name__ == "__main__":
    dataset_folder = global_helper.parse_args(global_helper.global_parser()).folder
    drums_df, dataset_folder = global_helper.load_dataset(dataset_folder,
                                                          dataset_filename=DATASET_WITH_FEATURES_FILENAME)
    run(drums_df, dataset_folder, "rf")
