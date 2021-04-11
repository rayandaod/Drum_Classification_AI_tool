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
    'random_forest': RandomForestClassifier(n_estimators=400, min_samples_split=2),
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
    return model


def split_data(drums_df):
    drums_df = drums_df.groupby('drum_type').head(params.N_SAMPLES_PER_CLASS)
    drum_type_labels, unique_labels = pd.factorize(drums_df.drum_type)
    drums_df = drums_df.assign(drum_type_labels=drum_type_labels)
    logger.info(f'Model output can be decoded with the following order of drum types: {list(unique_labels.values)}')
    logger.info(drums_df.info())

    train_clips_df, val_clips_df = train_test_split(drums_df, random_state=0)
    logger.info(f'{len(train_clips_df)} training sounds, {len(val_clips_df)} validation sounds')

    train_df = train_clips_df.drop(labels='drum_type_labels', axis=1)
    train_df = train_df.drop(labels='drum_type', axis=1)
    train_np = train_df.to_numpy()

    test_df = val_clips_df.drop(labels='drum_type_labels', axis=1)
    test_df = test_df.drop(labels='drum_type', axis=1)
    test_np = test_df.to_numpy()

    # There are occasionally random gaps in descriptors, so use imputation to fill in all values
    try:
        imp = pickle.load(open(params.IMPUTATER_PATH, 'rb'))
    except FileNotFoundError:
        logger.info(f'No cached inputer found, training')
        imp = IterativeImputer(max_iter=25, random_state=0)
        imp.fit(train_np)
        pickle.dump(imp, open(params.IMPUTATER_PATH, 'wb'))
    train_np = imp.transform(train_np)
    test_np = imp.transform(test_np)

    scaler = preprocessing.StandardScaler().fit(train_np)
    train_np = scaler.transform(train_np)
    test_np = scaler.transform(test_np)
    pickle.dump(scaler, open(params.SCALER_PATH, 'wb'))

    return train_np, train_clips_df.drum_type_labels, test_np, val_clips_df.drum_type_labels, list(unique_labels.values)


if __name__ == "__main__":
    drums_df = pd.read_pickle(params.PICKLE_DATASET_WITH_FEATURES_PATH)
    model = train(drums_df, "random_forest")
