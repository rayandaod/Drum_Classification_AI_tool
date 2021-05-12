import pickle
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from src import *
from config import TrainingConfig, GlobalConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prep_data_b4_training(data_prep_config, drums_df, dataset_folder):
    # Rather we cap the number of samples per class or not
    if data_prep_config.N_SAMPLES_PER_CLASS is not None:
        drums_df_caped = drums_df.groupby('drum_type').head(data_prep_config.N_SAMPLES_PER_CLASS)
    else:
        drums_df_caped = drums_df

    drum_type_labels, unique_labels = pd.factorize(drums_df_caped.drum_type)
    drums_df_labeled = drums_df_caped.assign(drum_type_labels=drum_type_labels)

    train_clips_df, val_clips_df = train_test_split(drums_df_labeled, random_state=GlobalConfig.RANDOM_STATE)
    logger.info(f'{len(train_clips_df)} training samples, {len(val_clips_df)} validation samples')

    # Remove last two columns (drum_type_labels and drum_type) for training
    columns_to_drop = ['drum_type_labels', 'drum_type']
    train_np = drop_columns(train_clips_df, columns_to_drop)
    test_np = drop_columns(val_clips_df, columns_to_drop)

    # There are occasionally random gaps in descriptors, so use imputation to fill in all values
    train_np, test_np = imputer(train_np, test_np, dataset_folder)

    # Standardize features by removing the mean and scaling to unit variance
    train_np, test_np = scaler(train_np, test_np, dataset_folder)

    print(list(unique_labels.values))
    return train_np, train_clips_df.drum_type_labels, test_np, val_clips_df.drum_type_labels, list(unique_labels.values)


def drop_columns(df, columns):
    for col in columns:
        df = df.drop(labels=col, axis=1)
    return df.to_numpy()


# Use imputation to fill in all missing values
def imputer(train_np, test_np, dataset_folder):
    try:
        imp = pickle.load(open(DATA_PATH / dataset_folder / IMPUTATER_FILENAME, 'rb'))
    except FileNotFoundError:
        logger.info(f'No cached imputer found, training')
        imp = TrainingConfig.SimpleTrainingConfig.iterative_imputer
        imp.fit(train_np)
        pickle.dump(imp, open(PICKLE_DATASETS_PATH/ dataset_folder / IMPUTATER_FILENAME, 'wb'))
    train_np = imp.transform(train_np)
    test_np = imp.transform(test_np)
    return train_np, test_np


# Standardize the training and testing sets by removing the mean and scaling to unit variance
def scaler(train_np, test_np, dataset_folder):
    scaler = preprocessing.StandardScaler().fit(train_np)
    train_np = scaler.transform(train_np)
    test_np = scaler.transform(test_np)
    pickle.dump(scaler, open(PICKLE_DATASETS_PATH / dataset_folder / SCALER_FILENAME, 'wb'))
    return train_np, test_np


# TODO: make sure these are the right classes
def get_class_distribution(obj):
    count_dict = {
        "kick": 0,
        "snare": 0,
        "hat": 0,
        "tom": 0
    }

    for i in obj:
        if i == 0:
            count_dict['hat'] += 1
        elif i == 1:
            count_dict['tom'] += 1
        elif i == 2:
            count_dict['snare'] += 1
        elif i == 3:
            count_dict['kick'] += 1
        else:
            print("Check classes.")

    print(count_dict)

    return count_dict


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc
