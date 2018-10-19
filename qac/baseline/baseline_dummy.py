"""Classifier that always predicts the most popular class.
"""
import argparse
import logging
import logging.config
import os
from os.path import dirname, exists, join

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder

from qac.evaluation import evaluation


def make_predictions(train_csv, test_csv, args):
    train_df = pd.read_csv(train_csv)
    X_train, y_train = train_df['id'], train_df['label']

    test_df = pd.read_csv(test_csv)
    X_test = test_df['id']

    dummy_classifier = DummyClassifier(strategy=args.strategy)
    dummy_classifier.fit(X_train.values.reshape(-1, 1), y_train)

    y_pred = dummy_classifier.predict(X_test.values.reshape(-1, 1))
    # select probability of UNCLEAR class which is considered to be the positive class
    y_pred_proba = dummy_classifier.predict_proba(X_test.values.reshape(-1, 1))[:, 1]

    return np.stack([X_test, y_pred, y_pred_proba], axis=1)


def save_predictions(pred, out_dir, run_id):
    os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame.from_dict({'id': pred[:, 0], 'y_pred': pred[:, 1], 'y_pred_proba': pred[:, 2]}) \
        .to_csv(join(out_dir, '{}_test.csv'.format(run_id)), index=False)


def run_exists(out_dir, run_id):
    return exists(join(out_dir, '{}_test.csv'.format(run_id)))


def main(args):
    in_dir = join(dirname(__file__), '../../data/labeled/')
    out_dir = join(dirname(__file__), '../../output/predictions/{}'.format(args.community))

    if not run_exists(out_dir, args.run_id):
        pred = make_predictions(
            join(in_dir, '{}_train.csv'.format(args.community)),
            join(in_dir, '{}_test.csv'.format(args.community)),
            args
        )
        save_predictions(pred, out_dir, args.run_id)

    le = LabelEncoder().fit(['CLEAR', 'UNCLEAR'])

    results_test = evaluation.evaluate(
        csv_true=join(in_dir, '{}_test.csv'.format(args.community)),
        csv_pred=join(out_dir, '{}_test.csv'.format(args.run_id)),
        label_encoder=le
    )
    evaluation.print_results(results_test, '{}_test'.format(args.run_id))


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("community", help="Community name", type=str)
    parser.add_argument("run_id", help="Run identifier", type=str)
    parser.add_argument("--strategy",
                        help="Dummy classifier strategy (default: most_frequent)",
                        type=str,
                        default='most_frequent')
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()
    log_dir = join(dirname(__file__), '../../output/predictions/{}'.format(args.community))
    os.makedirs(log_dir, exist_ok=True)
    log_filename = join(log_dir, '{}.log'.format(args.run_id))
    logging.config.fileConfig(join(dirname(__file__), '../logging_file.ini'),
                              disable_existing_loggers=False,
                              defaults={'logfilename': log_filename})
    main(arg_parser())
