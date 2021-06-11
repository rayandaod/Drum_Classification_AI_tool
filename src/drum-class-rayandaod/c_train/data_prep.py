import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from config import *
from c_train import helper
from z_helpers.paths import *
from z_helpers import global_helper


def prep_data_b4_training(drums_df, dataset_folder):
    logger.info("Preparing data...")

    # Create the data_prep_config
    data_prep_config = DataPrepConfig(os.path.basename(os.path.normpath(dataset_folder)))

    # Cap the number of samples per class or not
    drums_df = cap_or_not_cap(drums_df, data_prep_config)

    # TODO: is this really useful?
    drum_type_labels, unique_labels = pd.factorize(drums_df.drum_type)
    drums_df_labeled = drums_df.assign(drum_type_labels=drum_type_labels)

    # Split the data into a training and a testing set
    train_clips_df, val_clips_df = train_test_split(drums_df_labeled, random_state=GlobalConfig.RANDOM_STATE)
    logger.info(f'{len(train_clips_df)} training samples, {len(val_clips_df)} validation samples')

    columns_to_drop = ['drum_type_labels', 'drum_type', 'melS']
    train_np = global_helper.drop_columns(train_clips_df, columns_to_drop).to_numpy()
    test_np = global_helper.drop_columns(val_clips_df, columns_to_drop).to_numpy()

    return train_np, train_clips_df.drum_type_labels, test_np, val_clips_df.drum_type_labels, list(
        unique_labels.values), data_prep_config


def prep_data_b4_training_CNN(drums_df, dataset_folder):
    logger.info("Preparing data...")

    # Create the data_prep_config
    data_prep_config = DataPrepConfig(os.path.basename(os.path.normpath(str(dataset_folder))), isCNN=True)

    # Cap the number of samples per class or not
    drums_df = cap_or_not_cap(drums_df, data_prep_config)

    drum_type_labels, unique_labels = pd.factorize(drums_df.drum_type)
    target_feature = 'drum_type_labels'
    drums_df_labeled = drums_df.assign(drum_type_labels=drum_type_labels)

    train_clips_df, test_clips_df = train_test_split(drums_df_labeled, random_state=GlobalConfig.RANDOM_STATE)
    logger.info(f'{len(train_clips_df)} training samples, {len(test_clips_df)} validation samples')

    # Only keep the melS column and the labels
    train_clips_df = train_clips_df[['melS', 'drum_type_labels']]
    train_clips_flat = np.hstack(train_clips_df['melS'].to_numpy())

    test_clips_df = test_clips_df[['melS', 'drum_type_labels']]
    test_clips_flat = np.hstack(test_clips_df['melS'].to_numpy())

    # Create instances of ClipsDataset
    # TODO: Make sure we remove the mean and divide by std for each column
    train_dataset = helper.ClipsDataset(train_clips_df, 'mel_spec_model_input', target_feature,
                                        train_clips_flat.mean(), train_clips_flat.std())
    test_dataset = helper.ClipsDataset(test_clips_df, 'mel_spec_model_input', target_feature, test_clips_flat.mean(),
                                       test_clips_flat.std())

    # Create instances of Dataloader
    train_loader = helper.load(train_dataset, batch_size=data_prep_config.CNN_BATCH_SIZE, is_train=True,
                               desired_len=CNN_INPUT_SIZE[1])
    test_loader = helper.load(test_dataset, batch_size=data_prep_config.CNN_VAL_BATCH_SIZE, is_train=False,
                              desired_len=CNN_INPUT_SIZE[1])

    return train_loader, test_loader, list(unique_labels.values), data_prep_config


# Cap the number of samples per class or not
def cap_or_not_cap(drums_df, data_prep_config):
    if data_prep_config.N_SAMPLES_PER_CLASS is not None:
        print(drums_df)
        drums_df = drums_df.groupby('drum_type').head(data_prep_config.N_SAMPLES_PER_CLASS)
    return drums_df


def impute_and_scale(train_X, test_X, dataset_folder):
    # Use imputation to fill in all missing values
    def impute(train_np, test_np, dataset_folder):
        try:
            imp = pickle.load(open(DATA / dataset_folder / IMPUTATER_FILENAME, 'rb'))
        except FileNotFoundError:
            logger.info(f'No cached imputer found, training')
            imp = IterativeImputer(max_iter=TrainingConfig.Basic.IMP_MAX_ITER,
                                   random_state=GlobalConfig.RANDOM_STATE)
            imp.fit(train_np)
            pickle.dump(imp, open(DATASETS_PATH / dataset_folder / IMPUTATER_FILENAME, 'wb'))
        train_np = imp.transform(train_np)
        test_np = imp.transform(test_np)
        return train_np, test_np

    # Standardize the training and testing sets by removing the mean and scaling to unit variance
    def scale(train_np, test_np, dataset_folder):
        try:
            scaler = pickle.load(open(DATA / dataset_folder / SCALER_FILENAME, 'rb'))
        except FileNotFoundError:
            logger.info(f'No cached scaler found, scaling')
            scaler = preprocessing.StandardScaler().fit(train_np)
            pickle.dump(scaler, open(DATASETS_PATH / dataset_folder / SCALER_FILENAME, 'wb'))
        train_np = scaler.transform(train_np)
        test_np = scaler.transform(test_np)
        return train_np, test_np

    # There are occasionally random gaps in descriptors, so use imputation to fill in all values
    train_X, test_X = impute(train_X, test_X, dataset_folder)

    # Standardize features by removing the mean and scaling to unit variance
    train_X, test_X = scale(train_X, test_X, dataset_folder)

    return train_X, test_X
