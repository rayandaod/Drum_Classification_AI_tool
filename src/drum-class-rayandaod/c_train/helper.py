import torch
import operator
from torch.utils.data import Dataset
from functools import partial
from torch.utils.data import DataLoader

from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class ClipsDataset(Dataset):
    def __init__(self, clips_df, training_data_key, target_feature, mean, std):
        # Ensure the df has an index from 0 to len-1
        self.clips_df = clips_df.reset_index(drop=True)
        self.training_data_key = training_data_key
        self.target_feature = target_feature
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.clips_df)

    def __getitem__(self, index):
        clip = self.clips_df.loc[index]

        # Load from disk, normalize
        clip_feature = clip["melS"]
        clip_feature = np.divide(np.subtract(clip_feature, self.mean), self.std)

        # pytorch expects 3 dimensions (an extra one for channel) so wrap it
        if len(clip_feature.shape) == 2:
            clip_feature = np.expand_dims(clip_feature, 0)

        return clip_feature.astype(np.float32), clip[self.target_feature]


def collate(batch, desired_len):
    '''
    Pads the shorter inputs of a batch so they all have the same shape.
    :param max_length: Don't let inputs go beyond this size.
    :return: Batched inputs, a tensor of floats of shape [batch_size, 1, n_mels, length]
    '''
    # Inputs are [1 x n_mels x n_frames]
    n_mels = batch[0][0].shape[1]
    tensor = torch.zeros((len(batch), 1, n_mels, desired_len), dtype=torch.float, requires_grad=False)

    for batch_i, (instance, target) in enumerate(batch):
        replace_len = min(instance.shape[2], desired_len)
        trimmed_instance = instance[:, :, :replace_len]
        tensor.data[batch_i, :, :, :replace_len] = torch.FloatTensor(trimmed_instance)

    return tensor, torch.LongTensor(list(map(operator.itemgetter(1), batch)))


def load(seq_dataset, batch_size, is_train, desired_len, num_workers=8):
    return DataLoader(seq_dataset, batch_size=batch_size, shuffle=is_train,
                      collate_fn=partial(collate, desired_len=desired_len), num_workers=num_workers, pin_memory=True)


class ClassifierDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


def set_handler(handler):
    logger.addHandler(handler)