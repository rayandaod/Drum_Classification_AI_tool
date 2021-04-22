import logging
from argparse import ArgumentParser

from config import TrainingConfig, GlobalConfig
import preprocessing
import feature_engineering
import training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--reload', action='store_true', dest='reload',
                        help='Reload the drum library and extract the features again.')
    parser.add_argument('--verbose', action='store_true', dest='verbose',
                        help='Print useful data for debugging')
    parser.add_argument('--sweep', type=str, default=None, choices=TrainingConfig.grid_searches.keys(),
                        help='Run a gridSearch based on the hyper-parameters in config.py')
    parser.add_argument('--seed', type=int, default=None,
                        help='Set up a random state to have the same train-test-datasets across different runs')
    parser.set_defaults(reload=False)
    parser.set_defaults(verbose=False)
    args = parser.parse_args()

    GlobalConfig.RANDOM_STATE = args.seed
    logger.info(f"Random state = {GlobalConfig.RANDOM_STATE}")

    drums_df = preprocessing.load_drums_df(reload=args.reload, verbose=args.verbose)
    # TODO: add an optional or automatic use of data augmentation here + take into account the different folders after
    drums_df = feature_engineering.extract_all(drums_df, reload=args.reload, verbose=args.verbose)
    training.train(drums_df=drums_df, model_key="random_forest", grid_search_key=args.sweep)
