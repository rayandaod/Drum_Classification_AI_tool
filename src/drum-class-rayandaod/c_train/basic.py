import os
import sys
from sklearn.metrics import classification_report

sys.path.append(os.path.abspath(os.path.join('')))

from z_helpers import global_helper
from c_train import train_helper
from z_helpers.paths import *
from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fit_and_predict(model, train_X, train_y, test_X, test_y, drum_class_labels):
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    logger.info(classification_report(test_y, pred, target_names=drum_class_labels, zero_division=0))
    return model, test_X, test_y, drum_class_labels


def train(drums_df, model_key, dataset_folder, grid_search_key=None):
    data_prep_config = DataPrepConfig()
    train_X, train_y, test_X, test_y, drum_class_labels = train_helper.prep_data_b4_training(data_prep_config,
                                                                                             drums_df,
                                                                                             dataset_folder)
    logger.info(f"{model_key}:")

    # grid_search_key is already checked to be part of the grid_search_dict (in TrainingConfig) when launching main.py
    # it is therefore either None, or an existing key at this point
    if grid_search_key is not None:
        grid_search_dict = TrainingConfig.SimpleTrainingConfig.grid_searches
        # TODO: only for rf here, must be more general for other classification models
        for n in grid_search_dict[grid_search_key]["n_est_values"]:
            logger.info(f"{n} estimators:")
            model = RandomForestClassifier(n_estimators=n)
            fit_and_predict(model, train_X, train_y, test_X, test_y, drum_class_labels)
    else:
        model = TrainingConfig.SimpleTrainingConfig.MODELS[model_key]
        return fit_and_predict(model, train_X, train_y, test_X, test_y, drum_class_labels)


if __name__ == "__main__":
    drums_df, dataset_folder = global_helper.load_dataset(
        global_helper.parse_args(global_helper.global_parser()).folder, dataset_filename=DATASET_WITH_FEATURES_FILENAME)
    train(drums_df, "random_forest", dataset_folder)
