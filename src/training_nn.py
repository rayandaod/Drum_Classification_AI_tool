import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import helper
from config import PathConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    # x represents our data
    def forward(self, x):
        # Pass data through conv1
        x = self.conv1(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        # Run max pooling over x
        x = F.max_pool2d(x, 2)
        # Pass data through dropout1
        x = self.dropout1(x)
        # Flatten x with start_dim=1
        x = torch.flatten(x, 1)
        # Pass data through fc1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        # Apply softmax to x
        output = F.log_softmax(x, dim=1)
        return output


def fit_and_predict(train_X, train_y, test_X, test_y, drum_class_labels):
    my_nn = Net()
    result = my_nn(drums_df)
    return


def train(drums_df):
    train_X, train_y, test_X, test_y, drum_class_labels = helper.prepare_data(drums_df)
    logger.info("model: Deep NN")
    return fit_and_predict(train_X, train_y, test_X, test_y, drum_class_labels)


if __name__ == "__main__":
    drums_df = pd.read_pickle(PathConfig.PICKLE_DATASET_WITH_FEATURES_PATH)
    model, test_X, test_Y, labels = train(drums_df)
