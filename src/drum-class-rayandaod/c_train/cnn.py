import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import time
import shutil
from tqdm import tqdm
from torch.optim import SGD
from tensorboardX import SummaryWriter
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Accuracy, Loss
from ignite.handlers import EarlyStopping

from c_train import train_helper as helper
from config import *


class ConvNet(nn.Module):
    # Designed to work with the input size specified in __init__.py's CNN_INPUT_SIZE
    def __init__(self, cnn_training_config, n_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(12, 4), stride=2)
        self.conv1_batch = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=(4, 4), stride=(1, 2))
        self.conv2_batch = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, n_classes)  # x is a tensor object of size 1x128x259

    # `tensor` is size 1x128x259
    def forward(self, tensor, softmax=True):
        tensor = F.leaky_relu(
            F.max_pool2d(
                self.conv1_batch(self.conv1(tensor)),
                (4, 4)
            )
        )
        logger.debug(tensor.shape)
        tensor = F.leaky_relu(
            F.max_pool2d(
                self.conv2_batch(self.conv2(tensor)),
                (4, 8)
            )
        )
        logger.debug(tensor.shape)

        # Done with convolutions; two fully connected layers, and a softmax
        assert np.prod(tensor.shape[1:]) == 512
        tensor = tensor.view(-1, 512)
        tensor = F.leaky_relu(self.fc1(tensor))
        tensor = F.dropout(tensor, training=self.training)
        tensor = self.fc2(tensor)

        return F.log_softmax(tensor, dim=-1) if softmax else tensor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
here = Path(__file__).parent


def set_handler(handler):
    logger.addHandler(handler)


# To resume training of an existing model
# Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
def _load_checkpoint(model, optimizer, experiment_name):
    filename = MODELS_PATH / f'model_latest_{experiment_name}.pt'
    logger.info(f'Loading previous model state {filename}')
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        start_best_metric = checkpoint['best_accuracy']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        raise ValueError('No checkpoint found')

    return model, optimizer, start_epoch, start_best_metric


def train(model, train_loader, val_loader, epochs, early_stopping, lr, momentum, log_interval, experiment_name,
          continueing=False):
    # Choose the right device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    # Setup the summary writer
    writer = SummaryWriter(MODELS_PATH / f'tensorboard/runs_{experiment_name}')
    data_loader_iter = iter(train_loader)
    x, y = next(data_loader_iter)
    writer.add_graph(model, x)

    # Create the optimizer
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch = 0
    start_best_accuracy = 0.0

    if continueing:
        model, optimizer, start_epoch, start_best_accuracy = _load_checkpoint(model, optimizer, experiment_name)
        model.train()  # In case the model was saved after a test loop where model.eval() was called

    evaluator = create_supervised_evaluator(model, device=device,
                                            metrics={'accuracy': Accuracy(), 'nll': Loss(F.nll_loss)})
    evaluator_val = create_supervised_evaluator(model, device=device,
                                                metrics={'accuracy': Accuracy(), 'nll': Loss(F.nll_loss)})
    trainer = create_supervised_trainer(model, optimizer, nn.NLLLoss(), device=device)

    desc = 'ITERATION - loss: {:.4f}'
    progress_bar = tqdm(
        initial=0, leave=False, total=len(train_loader),
        desc=desc.format(0)
    )

    @trainer.on(Events.STARTED)
    def init(engine):
        engine.state.epoch = start_epoch
        engine.state.best_accuracy = start_best_accuracy

    # One iteration = one batch
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % log_interval == 0:
            progress_bar.desc = desc.format(engine.state.output)
            progress_bar.update(log_interval)
            writer.add_scalar('training/loss', engine.state.output, engine.state.iteration)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_gradients(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % log_interval == 0:
            for n, p in model.named_parameters():
                if p.requires_grad:
                    writer.add_scalar(f'{n}/gradient', p.grad.abs().mean(), engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        progress_bar.refresh()
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        logger.info(
            'Training Results - Epoch: {}  Avg accuracy: {:.4f} Avg loss: {:.4f}'.format(engine.state.epoch,
                                                                                         avg_accuracy, avg_nll)
        )
        writer.add_scalar('training/avg_loss', avg_nll, engine.state.epoch)
        writer.add_scalar('training/avg_accuracy', avg_accuracy, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results_and_save(engine):
        evaluator_val.run(val_loader)
        metrics = evaluator_val.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        logger.info(
            'Validation Results - Epoch: {}  Avg accuracy: {:.4f} Avg loss: {:.4f}'.format(engine.state.epoch,
                                                                                           avg_accuracy, avg_nll))

        progress_bar.n = progress_bar.last_print_n = 0
        writer.add_scalar('validation/avg_loss', avg_nll, engine.state.epoch)
        writer.add_scalar('validation/avg_accuracy', avg_accuracy, engine.state.epoch)

        # Save the model every epoch. If it's the best seen so far, save it separately
        torch.save({
            'epoch': engine.state.epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': avg_accuracy,
            'best_accuracy': engine.state.best_accuracy,
            'loss': avg_nll
        }, f'model_latest_{experiment_name}.pt')
        if avg_accuracy > engine.state.best_accuracy:
            engine.state.best_accuracy = avg_accuracy
            shutil.copyfile(f'model_latest_{experiment_name}.pt', f'model_best_{experiment_name}.pt')

    # Early stopping
    handler = EarlyStopping(patience=early_stopping,
                            score_function=(lambda engine: -evaluator_val.state.metrics['nll']), trainer=trainer)
    evaluator_val.add_event_handler(Events.COMPLETED, handler)

    trainer.run(train_loader, max_epochs=epochs)
    progress_bar.close()
    writer.close()


def run(drums_df, exp_name, is_continue):
    # PREPARE DATA
    data_prep_config = DataPrepConfig()
    train_loader, test_loader, _ = helper.prep_data_b4_training_CNN(data_prep_config, drums_df)

    # CREATE THE MODEL
    cnn_training_config = TrainingConfig.CNNTrainingConfig()
    model = ConvNet(cnn_training_config, n_classes=len(GlobalConfig.DRUM_TYPES))

    # TRAIN
    train(model, train_loader, test_loader, cnn_training_config.MAX_EPOCHS, cnn_training_config.EARLY_STOPPING,
          cnn_training_config.LEARNING_RATE, cnn_training_config.MOMENTUM, cnn_training_config.LOG_INTERVAL,
          exp_name, continueing=is_continue)


if __name__ == "__main__":
    # Load the parser
    parser = global_parser()

    parser.add_argument('--continue_name', type=str,
                        help='allows you to continue previous training, given an experiment name. Look for a model_latest_[experiment].pt and training_[experiment].log to get the name. For now, epoch seconds is used')
    parser.add_argument('--eval', action='store_true',
                        help='Dont train, just load_drums the best model (must provide --continue_name) and print the accuracy')

    args = parse_args(parser)
    dataset_folder = args.folder

    # Load the dataset
    drums_df = pd.read_pickle(
        PICKLE_DATASETS_PATH / dataset_folder / DATASET_WITH_FEATURES_FILENAME)

    # Start the training
    experiment_name = args.continue_name if args.continue_name is not None else str(int(time.time()))
    continue_exp = args.continue_name is not None
    run(drums_df, experiment_name, is_continue=continue_exp)
