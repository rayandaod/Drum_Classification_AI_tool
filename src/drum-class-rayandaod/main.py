import sys
import os

sys.path.append(os.path.abspath(os.path.join('')))

from z_helpers import global_helper
from config import *
from b_features import extract
from a_data import load
from c_train import basic


def parse_more_arguments(parser):
    # Add more arguments to the loaded global parser
    parser.add_argument('--sweep', type=str, default=None,
                        choices=TrainingConfig.Basic.grid_searches.keys(),
                        help='Run a gridSearch based on the hyper-parameters in config.py')
    parser.add_argument('--seed', type=int, default=None,
                        help='Set up a random state to have the same train-test-datasets across different runs')
    parser.add_argument('--model', type=str, default="random_forest",
                        choices=TrainingConfig.Basic.MODELS,
                        help='Choose a model to train with')

    # Parse the global arguments and the new ones
    args = global_helper.parse_args(parser)

    # Set the new value for RANDOM_STATE
    GlobalConfig.RANDOM_STATE = args.seed
    logger.info(f"Random state = {GlobalConfig.RANDOM_STATE}")

    return args


if __name__ == "__main__":
    args = parse_more_arguments(global_helper.global_parser())
    _, dataset_folder = load.run_or_load(args.folder)
    drums_df = extract.run_or_load(dataset_folder)
    basic.train(drums_df, model_key=args.model, grid_search_key=args.sweep, dataset_folder=dataset_folder)
    # training_nn.run(drums_df, dataset_folder)
