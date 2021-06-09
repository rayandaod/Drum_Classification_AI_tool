import torch
import time
import os
import sys
import json
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torchsummary import summary

sys.path.append(os.path.abspath(os.path.join('')))

from c_train import helper
from config import *
from z_helpers import global_helper
from z_helpers.paths import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run(drums_df, dataset_folder):
    # PREPARE DATA
    data_prep_config = DataPrepConfig(os.path.basename(os.path.normpath(dataset_folder)))
    X_train, X_val, y_train, y_val, X_test, y_test = prepare_data(data_prep_config, drums_df, dataset_folder)

    # TRAIN
    nn_config = TrainingConfig.NN()
    nn_config.N_INPUT = X_train.shape[1]
    model = helper.MulticlassClassification(num_feature=nn_config.N_INPUT,
                                            num_class=len(GlobalConfig.DRUM_TYPES),
                                            nn_config=nn_config)
    model, logs_string = fit_and_predict(model, nn_config, X_train, y_train.to_numpy(), X_val, y_val.to_numpy(),
                                          X_test,
                                          y_test.to_numpy())

    # SAVE MODEL & METADATA
    save_model(model, data_prep_config, nn_config, logs_string)


def prepare_data(data_prep_config, drums_df, dataset_folder):
    logger.info("Preparing data...")
    X_trainval, y_trainval, X_test, y_test, _ = helper.prep_data_b4_training(data_prep_config, drums_df, dataset_folder)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval,
                                                      test_size=data_prep_config.VALIDATION_SET_RATIO,
                                                      stratify=y_trainval,
                                                      random_state=GlobalConfig.RANDOM_STATE)

    return X_train, X_val, y_train, y_val, X_test, y_test


def fit_and_predict(model, nn_config, train_X, train_y, val_X, val_y, test_X, test_y):
    logger.info(f"Training with {model.name}...")

    train_dataset = helper.ClassifierDataset(torch.from_numpy(train_X).float(), torch.from_numpy(train_y).long())
    val_dataset = helper.ClassifierDataset(torch.from_numpy(val_X).float(), torch.from_numpy(val_y).long())
    test_dataset = helper.ClassifierDataset(torch.from_numpy(test_X).float(), torch.from_numpy(test_y).long())

    target_list = []
    for _, t in train_dataset:
        target_list.append(t)
    target_list = torch.tensor(target_list)
    target_list = target_list[torch.randperm(len(target_list))]

    class_count = [i for i in helper.get_class_distribution(train_y).values()]
    class_weights = 1. / torch.tensor(class_count, dtype=torch.float)

    class_weights_all = class_weights[target_list]

    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True
    )

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=nn_config.BATCH_SIZE,
                              sampler=weighted_sampler
                              )
    val_loader = DataLoader(dataset=val_dataset, batch_size=1)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=nn_config.LEARNING_RATE)

    print(model)

    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }

    logger.info("Begin training...")
    logs_string = ''
    for e in range(1, nn_config.EPOCHS + 1):
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch)

            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = helper.multi_acc(y_train_pred, y_train_batch)

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        # VALIDATION
        with torch.no_grad():

            val_epoch_loss = 0
            val_epoch_acc = 0

            model.eval()
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                y_val_pred = model(X_val_batch)

                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = helper.multi_acc(y_val_pred, y_val_batch)

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

        loss_stats['train'].append(train_epoch_loss / len(train_loader))
        loss_stats['val'].append(val_epoch_loss / len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc / len(val_loader))

        logs_string = global_helper.print_and_append(logs_string,
            f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | Val Loss: {val_epoch_loss / len(val_loader):.5f} | Train Acc: {train_epoch_acc / len(train_loader):.3f}| Val Acc: {val_epoch_acc / len(val_loader):.3f}')

    # TEST
    y_pred_list = []
    with torch.no_grad():
        model.eval()
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            _, y_pred_tags = torch.max(y_test_pred, dim=1)
            y_pred_list.append(y_pred_tags.cpu().numpy())
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

    classification_report_string = 'Classification Report:\n' + classification_report(test_y, y_pred_list)
    print(classification_report_string)

    return model, classification_report_string + logs_string


def save_model(model, data_prep_config, nn_config, logs_string):
    logger.info("Saving model and metadata...")

    # Create the model folder
    model_folder = time.strftime("%Y%m%d-%H%M%S")
    model_folder_path = MODELS / model_folder
    os.makedirs(model_folder_path)

    # Save the model
    torch.save(model, model_folder_path / MODEL_FILENAME)

    # Save the metadata
    metadata = {
        "model_name": model.name,
        "training_params": json.dumps(data_prep_config.__dict__),
        "NN_training params": json.dumps(nn_config.__dict__),
    }
    with open(model_folder_path / METADATA_JSON_FILENAME, 'w') as outfile:
        json.dump(metadata, outfile)

    # Save the logs
    log_file = open(model_folder_path / LOGS_FILENAME, "a+")
    log_file.write(logs_string)
    log_file.close()


if __name__ == "__main__":
    dataset_folder = global_helper.parse_args(global_helper.global_parser()).folder
    drums_df, dataset_folder = global_helper.load_dataset(dataset_folder,
                                                          dataset_filename=DATASET_WITH_FEATURES_FILENAME)

    # Start the training
    run(drums_df, dataset_folder)
