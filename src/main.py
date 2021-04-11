from argparse import ArgumentParser

import preprocessing
import feature_engineering
import training

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--reload', action='store_true', dest='reload',
                        help='Reload the drum library and extract the features again.')
    parser.add_argument('--verbose', action='store_true', dest='verbose',
                        help='Print useful data for debugging')
    parser.set_defaults(reload=False)
    parser.set_defaults(verbose=False)
    args = parser.parse_args()

    drums_df = preprocessing.load_drums_df(reload=args.reload, verbose=args.verbose)
    drums_df = feature_engineering.extract_all(drums_df, reload=args.reload, verbose=args.verbose)
    training.train(drums_df=drums_df, model_key="random_forest", verbose=args.verbose)
