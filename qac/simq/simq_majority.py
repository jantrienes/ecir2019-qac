import argparse
import logging
import logging.config
import os
from os.path import dirname, exists, join

import numpy as np
import pandas as pd

import simq_features
from qac.evaluation import evaluation
from qac.experiments import preprocessing

logger = logging.getLogger(__name__)  # pylint: disable=locally-disabled, invalid-name

DEBUG = False


def make_predictions(args):
    test_ids = preprocessing.load_community(
        args.community, preprocess=False, with_dev=True).test_ids

    feature_handler = simq_features.FeatureHandler(args.community, run=args.simq_run)

    majority = feature_handler.read(feature_name='feat_all_majority_k10')
    fraction = feature_handler.read(feature_name='feat_all_fraction_k10')
    features = pd.concat([majority, fraction], axis=1)
    features['feat_all_fraction_unclear_k10'] = (1 - features.feat_all_fraction_k10)

    instances = features.loc[test_ids]

    y_pred = instances.feat_all_majority_k10.values
    y_pred_proba = instances.feat_all_fraction_unclear_k10.values  # probability is for class UNCLEAR

    return np.stack([test_ids, y_pred, y_pred_proba], axis=1)


def save_predictions(pred, out_dir, run_id):
    os.makedirs(out_dir, exist_ok=True)

    y_pred = pred[:, 1].astype(int)
    y_pred = preprocessing.LABEL_ENCODER.inverse_transform(y_pred)
    pd.DataFrame.from_dict({'id': pred[:, 0], 'y_pred': y_pred, 'y_pred_proba': pred[:, 2]}) \
        .to_csv(join(out_dir, '{}_test.csv'.format(run_id)), index=False)


def run_exists(out_dir, run_id):
    return exists(join(out_dir, '{}_test.csv'.format(run_id)))


def main(args):
    in_dir = join(dirname(__file__), '../../data/labeled/')
    out_dir = join(dirname(__file__), '../../output/predictions/{}'.format(args.community))

    if not run_exists(out_dir, args.run_id) or DEBUG:
        logger.info('Run %s does not exists. Start new...', args.run_id)
        pred = make_predictions(args)
        save_predictions(pred, out_dir, args.run_id)
        logger.info('Finish training and testing.')

    results_test = evaluation.evaluate(
        csv_true=join(in_dir, '{}_test.csv'.format(args.community)),
        csv_pred=join(out_dir, '{}_test.csv'.format(args.run_id)),
        label_encoder=preprocessing.LABEL_ENCODER
    )
    evaluation.print_results(results_test, '{}_test'.format(args.run_id))


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("community", help="Community name", type=str)
    parser.add_argument("run_id", help="Run identifier", type=str)
    parser.add_argument("simq_run", help="Identifier of similar question retrieval run", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()
    log_dir = join(dirname(__file__), '../../output/predictions/{}'.format(args.community))
    os.makedirs(log_dir, exist_ok=True)
    log_filename = join(log_dir, '{}.log'.format(args.run_id))
    logging.getLogger('elasticsearch').setLevel(logging.WARNING)
    logging.config.fileConfig(join(dirname(__file__), '../logging_file.ini'),
                              disable_existing_loggers=False,
                              defaults={'logfilename': log_filename})
    main(args)
