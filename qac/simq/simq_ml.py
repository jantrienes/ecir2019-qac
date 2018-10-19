"""Logistic regression with tf-idf weighting. Optimal parameter settings are found via grid search.
"""
import argparse
import logging
import logging.config
import os
from os.path import dirname, exists, join

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qac.evaluation import evaluation
from qac.experiments import preprocessing
from qac.simq.simq_features import FeatureHandler

logger = logging.getLogger(__name__)  # pylint: disable=locally-disabled, invalid-name

feat_group1 = [
    'feat_post_length',
    'feat_post_contains_preformatted',
    'feat_post_contains_blockquote',
    'feat_post_contains_questionmark',
    'feat_post_readability'
]

feat_group2 = [
    'feat_all_sim_sum',
    'feat_all_sim_max',
    'feat_all_sim_avg',
    'feat_all_num_similar_k10',
    'feat_all_num_similar_k20',
    'feat_all_num_similar_k50',
    'feat_all_num_clear_k10',
    'feat_all_num_clear_k20',
    'feat_all_num_clear_k50',
    'feat_all_num_unclear_k10',
    'feat_all_num_unclear_k20',
    'feat_all_num_unclear_k50',
    'feat_all_majority_k10',
    'feat_all_majority_k20',
    'feat_all_majority_k50',
    'feat_all_ratio_k10',
    'feat_all_ratio_k20',
    'feat_all_ratio_k50',
    'feat_all_fraction_k10',
    'feat_all_fraction_k20',
    'feat_all_fraction_k50',
]

feat_group3 = [
    'feat_unclear_global_cos',
    'feat_unclear_individual_cos',
    'feat_unclear_individual_cos_weighted'
]

FEATURE_GROUPS = {
    'all': feat_group1 + feat_group2 + feat_group3,
    '1': feat_group1,
    '2': feat_group2,
    '3': feat_group3
}

def feature_names(groups):
    features = []
    for group_name in groups:
        features += FEATURE_GROUPS[group_name]
    return features


def make_predictions(args):
    community = preprocessing.load_community(args.community, with_dev=True)
    fh = FeatureHandler(args.community, args.simq_run)

    df = fh.read_all()
    df = df * 1  # convert boolean features to ints
    df = df[feature_names(args.feature_groups)]

    X_train, y_train = df.loc[community.train_ids], community.y_train
    X_val, y_val = df.loc[community.val_ids], community.y_val
    X_test = df.loc[community.test_ids]

    pipeline = Pipeline([
        ('scaling', StandardScaler()),
        ('clf', LogisticRegression(C=1))
    ])

    # no hyperparameter tuning, so we can combine training and dev sets
    pipeline.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))

    logger.info('Test simq classifier...')
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    return np.stack([community.test_ids, y_pred, y_pred_proba], axis=1)


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

    if not run_exists(out_dir, args.run_id):
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
    parser.add_argument("--feature_groups",
                        help="Feature groups",
                        type=str,
                        default=['all'],
                        choices=['all', '1', '2', '3'],
                        nargs='*')
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()
    log_dir = join(dirname(__file__), '../../output/predictions/{}'.format(args.community))
    os.makedirs(log_dir, exist_ok=True)
    log_filename = join(log_dir, '{}.log'.format(args.run_id))
    logging.config.fileConfig(join(dirname(__file__), '../logging_file.ini'),
                              disable_existing_loggers=False,
                              defaults={'logfilename': log_filename})
    main(args)
