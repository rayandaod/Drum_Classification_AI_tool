import pickle
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import models.data_loader as data_loader
from config import *
from models.clips_dataset import ClipsDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prep_data_b4_training(data_prep_config, drums_df, dataset_folder):
    logger.info("Preparing data...")

    # Cap the number of samples per class or not
    drums_df = cap_or_not_cap(drums_df, data_prep_config)

    # TODO: is this really useful?
    drum_type_labels, unique_labels = pd.factorize(drums_df.drum_type)
    drums_df_labeled = drums_df.assign(drum_type_labels=drum_type_labels)

    # Split the data into a training and a testing set
    train_clips_df, val_clips_df = train_test_split(drums_df_labeled, random_state=GlobalConfig.RANDOM_STATE)
    logger.info(f'{len(train_clips_df)} training samples, {len(val_clips_df)} validation samples')

    # Remove the useless columns
    columns_to_drop = ['drum_type_labels', 'drum_type', 'melS']
    train_np = drop_columns(train_clips_df, columns_to_drop)
    test_np = drop_columns(val_clips_df, columns_to_drop)

    # There are occasionally random gaps in descriptors, so use imputation to fill in all values
    train_np, test_np = imputer(train_np, test_np, dataset_folder)

    # Standardize features by removing the mean and scaling to unit variance
    train_np, test_np = scaler(train_np, test_np, dataset_folder)

    return train_np, train_clips_df.drum_type_labels, test_np, val_clips_df.drum_type_labels, list(unique_labels.values)


def prep_data_b4_training_CNN(data_prep_config, drums_df):
    logger.info("Preparing data...")

    # Cap the number of samples per class or not
    drums_df = cap_or_not_cap(drums_df, data_prep_config)

    drum_type_labels, unique_labels = pd.factorize(drums_df.drum_type)
    target_feature = 'drum_type_labels'
    drums_df_labeled = drums_df.assign(drum_type_labels=drum_type_labels)

    train_clips_df, test_clips_df = train_test_split(drums_df_labeled, random_state=GlobalConfig.RANDOM_STATE)
    logger.info(f'{len(train_clips_df)} training samples, {len(test_clips_df)} validation samples')

    # Only keep the melS column and the labels
    train_clips_df = train_clips_df[['melS', 'drum_type_labels']]
    test_clips_df = test_clips_df[['melS', 'drum_type_labels']]

    # Create instances of ClipsDataset
    train_dataset = ClipsDataset(train_clips_df, 'mel_spec_model_input', target_feature,
                                 np.mean(train_clips_df['melS']), np.std(train_clips_df['melS']))
    test_dataset = ClipsDataset(test_clips_df, 'mel_spec_model_input', target_feature, np.mean(test_clips_df['melS']),
                                np.std(test_clips_df['melS']))

    # Create instances of Dataloader
    train_loader = data_loader.load(train_dataset, batch_size=data_prep_config.CNN_BATCH_SIZE, is_train=True,
                                    desired_len=CNN_INPUT_SIZE[1])
    test_loader = data_loader.load(test_dataset, batch_size=data_prep_config.CNN_VAL_BATCH_SIZE, is_train=False,
                                   desired_len=CNN_INPUT_SIZE[1])

    return train_loader, test_loader, list(unique_labels.values)


# Cap the number of samples per class or not
def cap_or_not_cap(drums_df, data_prep_config):
    if data_prep_config.N_SAMPLES_PER_CLASS is not None:
        drums_df = drums_df.groupby('drum_type').head(data_prep_config.N_SAMPLES_PER_CLASS)
    return drums_df


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
        pickle.dump(imp, open(PICKLE_DATASETS_PATH / dataset_folder / IMPUTATER_FILENAME, 'wb'))
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
