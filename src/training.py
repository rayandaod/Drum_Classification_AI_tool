import logging
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import helper
from config import TrainingConfig, PathConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

here = Path(__file__).parent


def fit_and_predict(model, train_X, train_y, test_X, test_y, drum_class_labels):
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    logger.info(classification_report(test_y, pred, target_names=drum_class_labels, zero_division=0))
    return model, test_X, test_y, drum_class_labels


def train(drums_df, model_key, grid_search_key=None):
    train_X, train_y, test_X, test_y, drum_class_labels = helper.prepare_data(drums_df)
    logger.info(f"{model_key}:")

    # grid_search_key is already checked to be part of the grid_search_dict (in TrainingConfig) when launching main.py
    # it is therefore either None, or an existing key at this point
    if grid_search_key is not None:
        grid_search_dict = TrainingConfig.grid_searches
        # TODO: only for rf here, must be more general for other classification models
        for n in grid_search_dict[grid_search_key]["n_est_values"]:
            logger.info(f"{n} estimators:")
            model = RandomForestClassifier(n_estimators=n)
            fit_and_predict(model, train_X, train_y, test_X, test_y, drum_class_labels)
    else:
        model = TrainingConfig.MODELS[model_key]
        return fit_and_predict(model, train_X, train_y, test_X, test_y, drum_class_labels)


if __name__ == "__main__":
    drums_df = pd.read_pickle(PathConfig.PICKLE_DATASET_WITH_FEATURES_PATH)
    model, test_X, test_Y, labels = train(drums_df, "random_forest")
