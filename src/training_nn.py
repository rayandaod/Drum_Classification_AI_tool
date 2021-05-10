import logging
import pandas as pd
import torch

import helper
from config import PathConfig
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EPOCHS = 300
BATCH_SIZE = 16
LEARNING_RATE = 0.0007


class ClassifierDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()

        self.layer_1 = nn.Linear(num_feature, 64)
        self.layer_2 = nn.Linear(64, 32)
        self.layer_3 = nn.Linear(32, 16)
        self.layer_out = nn.Linear(16, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.batchnorm3 = nn.BatchNorm1d(16)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x


def fit_and_predict(train_X, train_y, val_X, val_y, test_X, test_y, drum_class_labels):
    # print(train_y)
    train_dataset = ClassifierDataset(torch.from_numpy(train_X).float(), torch.from_numpy(train_y).long())
    val_dataset = ClassifierDataset(torch.from_numpy(val_X).float(), torch.from_numpy(val_y).long())
    test_dataset = ClassifierDataset(torch.from_numpy(test_X).float(), torch.from_numpy(test_y).long())

    target_list = []
    for _, t in train_dataset:
        target_list.append(t)

    target_list = torch.tensor(target_list)
    target_list = target_list[torch.randperm(len(target_list))]

    class_count = [i for i in get_class_distribution(train_y).values()]
    class_weights = 1. / torch.tensor(class_count, dtype=torch.float)

    class_weights_all = class_weights[target_list]

    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True
    )

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              sampler=weighted_sampler
                              )
    val_loader = DataLoader(dataset=val_dataset, batch_size=1)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MulticlassClassification(num_feature=train_X.shape[1], num_class=4)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(model)

    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }

    print("Begin training.")
    for e in tqdm(range(1, EPOCHS + 1)):
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch)

            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)

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
                val_acc = multi_acc(y_val_pred, y_val_batch)

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

    loss_stats['train'].append(train_epoch_loss / len(train_loader))
    loss_stats['val'].append(val_epoch_loss / len(val_loader))
    accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
    accuracy_stats['val'].append(val_epoch_acc / len(val_loader))

    print(
        f'Epoch {e + 0:03}: | Train Loss: {train_epoch_loss / len(train_loader):.5f} | Val Loss: {val_epoch_loss / len(val_loader):.5f} | Train Acc: {train_epoch_acc / len(train_loader):.3f}| Val Acc: {val_epoch_acc / len(val_loader):.3f}')


def get_class_distribution(obj):
    count_dict = {
        "kick": 0,
        "snare": 0,
        "hat": 0,
        "tom": 0
    }

    for i in obj:
        if i == 0:
            count_dict['kick'] += 1
        elif i == 1:
            count_dict['snare'] += 1
        elif i == 2:
            count_dict['hat'] += 1
        elif i == 3:
            count_dict['tom'] += 1
        else:
            print("Check classes.")

    return count_dict


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc


def train(drums_df, dataset_folder):
    X_trainval, y_trainval, test_X, test_y, drum_class_labels = helper.prepare_data(drums_df, dataset_folder)

    # Split train into train-val
    train_X, val_X, train_y, val_y = train_test_split(X_trainval, y_trainval, test_size=0.1, stratify=y_trainval,
                                                      random_state=21)
    logger.info("model: Deep NN")
    return fit_and_predict(train_X, train_y.to_numpy(), val_X, val_y.to_numpy(), test_X, test_y.to_numpy(), drum_class_labels)


if __name__ == "__main__":
    parser = helper.create_global_parser()
    args = helper.parse_global_arguments(parser)
    dataset_folder = args.old
    drums_df = pd.read_pickle(PathConfig.PICKLE_DATASETS_PATH / dataset_folder / PathConfig.DATASET_WITH_FEATURES_FILENAME)
    train(drums_df, dataset_folder)
