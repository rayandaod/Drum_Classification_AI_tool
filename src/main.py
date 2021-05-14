from config import *
from training import training
from data_to_features import data_augmentation, preprocessing, feature_engineering


def parse_arguments():
    # Load the global parser
    parser = global_parser()

    # Add more arguments to the loaded global parser
    parser.add_argument('--sweep', type=str, default=None, choices=TrainingConfig.SimpleTrainingConfig.grid_searches.keys(),
                        help='Run a gridSearch based on the hyper-parameters in config.py')
    parser.add_argument('--seed', type=int, default=None,
                        help='Set up a random state to have the same train-test-datasets across different runs')
    parser.add_argument('--model', type=str, default="random_forest", choices=TrainingConfig.SimpleTrainingConfig.MODELS,
                        help='Choose a model to train with')

    # Parse the global arguments and the new ones
    args = parse_args(parser)

    # Set the new value for RANDOM_STATE
    GlobalConfig.RANDOM_STATE = args.seed
    logger.info(f"Random state = {GlobalConfig.RANDOM_STATE}")

    return args


if __name__ == "__main__":
    args = parse_arguments()
    drums_df, dataset_folder = preprocessing.load_drums_df(dataset_folder=args.old)
    # TODO: add an optional or automatic use of data augmentation here + append to previous dataframe
    drums_df = data_augmentation.augment_data(drums_df, dataset_folder)  # useless for now
    drums_df = feature_engineering.extract_all(drums_df, dataset_folder=dataset_folder)
    training.train(drums_df, model_key=args.model, grid_search_key=args.sweep,
                   dataset_folder=dataset_folder)
