import logging
import pandas as pd
import pickle
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

here = Path(__file__).parent


MODELS = {
    'lr': LinearRegression(),
    'svc': SVC(),
    'random_forest': RandomForestClassifier(n_estimators=500),
    'gb': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=0),
    'knn': KNeighborsClassifier()
}


def train(drums_df, model_key, verbose=False):
    train_X, train_y, test_X, test_y, drum_class_labels = split_data(drums_df)
    model = MODELS[model_key]
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    logger.info(f"{model_key}:")
    logger.info(classification_report(test_y, pred, target_names=drum_class_labels, zero_division=0))
    return model, test_X, test_y, drum_class_labels


def split_data(drums_df):
    drums_df_caped = drums_df.groupby('drum_type').head(params.N_SAMPLES_PER_CLASS)
    drum_type_labels, unique_labels = pd.factorize(drums_df_caped.drum_type)
    drums_df_labeled = drums_df_caped.assign(drum_type_labels=drum_type_labels)

    logger.info(f'Model output can be decoded with the following order of drum types: {list(unique_labels.values)}')
    logger.info(drums_df_labeled.info())

    train_clips_df, val_clips_df = train_test_split(drums_df_labeled, random_state=0)
    logger.info(f'{len(train_clips_df)} training sounds, {len(val_clips_df)} validation sounds')

    # Remove last two columns (drum_type_labels and drum_type) for training
    columns_to_drop = ['drum_type_labels', 'drum_type']
    train_np = drop_columns(train_clips_df, columns_to_drop)
    test_np = drop_columns(val_clips_df, columns_to_drop)

    # There are occasionally random gaps in descriptors, so use imputation to fill in all values
    train_np, test_np = imputer(train_np, test_np)

    # Standardize features by removing the mean and scaling to unit variance
    train_np, test_np = scaler(train_np, test_np)

    return train_np, train_clips_df.drum_type_labels, test_np, val_clips_df.drum_type_labels, list(unique_labels.values)


def drop_columns(df, columns):
    for col in columns:
        df = df.drop(labels=col, axis=1)
    return df.to_numpy()


def imputer(train_np, test_np):
    try:
        imp = pickle.load(open(params.IMPUTATER_PATH, 'rb'))
    except FileNotFoundError:
        logger.info(f'No cached imputer found, training')
        imp = IterativeImputer(max_iter=25, random_state=0)
        imp.fit(train_np)
        pickle.dump(imp, open(params.IMPUTATER_PATH, 'wb'))
    train_np = imp.transform(train_np)
    test_np = imp.transform(test_np)
    return train_np, test_np


def scaler(train_np, test_np):
    scaler = preprocessing.StandardScaler().fit(train_np)
    train_np = scaler.transform(train_np)
    test_np = scaler.transform(test_np)
    pickle.dump(scaler, open(params.SCALER_PATH, 'wb'))
    return train_np, test_np


if __name__ == "__main__":
    drums_df = pd.read_pickle(params.PICKLE_DATASET_WITH_FEATURES_PATH)
    model, test_X, test_Y, labels = train(drums_df, "random_forest")
