"""Logistic regression with tf-idf weighting. Optimal parameter settings are found via grid search.
"""
import argparse
import logging
import logging.config
import os
from os.path import dirname, exists, join

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from qac.evaluation import evaluation
from qac.experiments import preprocessing

logger = logging.getLogger(__name__)  # pylint: disable=locally-disabled, invalid-name

def identity(x):
    return x

def make_predictions(args):
    X_train, y_train, X_test, _, _, test_ids = preprocessing.load_community(
        args.community, preprocess=True, min_df=3
    )

    pipeline = Pipeline([
        ('vect', TfidfVectorizer(ngram_range=(1, args.ngram_range),
            analyzer='word', tokenizer=identity, preprocessor=identity)),
        ('clf', LogisticRegression(C=1)),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

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

    if not run_exists(out_dir, args.run_id):
        logger.info('Start fitting logistic regression classifier')
        pred =  make_predictions(args)
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
    parser.add_argument("--ngram_range", help="Run identifier", type=int, default=1)
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()
    log_dir = join(dirname(__file__), '../../output/predictions/{}'.format(args.community))
    log_filename = join(log_dir, '{}.log'.format(args.run_id))
    logging.config.fileConfig(join(dirname(__file__), '../logging_file.ini'),
                              disable_existing_loggers=False,
                              defaults={'logfilename': log_filename})
    main(args)
