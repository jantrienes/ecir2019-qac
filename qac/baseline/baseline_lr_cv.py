"""Logistic regression with tf-idf weighting. Optimal parameter settings are found via grid search.
"""
import argparse
import json
import logging
import logging.config
import os
from os.path import dirname, exists, join

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from qac.experiments import preprocessing
from qac.evaluation import evaluation

logger = logging.getLogger(__name__)  # pylint: disable=locally-disabled, invalid-name

def identity(x):
    return x

def make_predictions(args):
    X_train, y_train, X_test, _, _, test_ids = preprocessing.load_community(
        args.community, preprocess=True, min_df=3
    )

    pipeline = Pipeline([
        ('vect', TfidfVectorizer(analyzer='word', tokenizer=identity, preprocessor=identity)),
        ('clf', LogisticRegression()),
    ])

    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'clf__C': [100, 10, 1, 0.1, 0.01]
    }

    logger.info('Start fitting logistic regression classifier...')
    gs = GridSearchCV(pipeline, parameters, n_jobs=args.n_jobs, cv=5, verbose=1,
                      scoring='roc_auc', return_train_score=True)
    gs.fit(X_train, y_train)

    logger.info('Test logistic regression classifier...')
    y_pred = gs.predict(X_test)
    y_pred_proba = gs.predict_proba(X_test)[:, 1]

    return np.stack([test_ids, y_pred, y_pred_proba], axis=1), gs.cv_results_, gs.best_params_


def save_predictions(pred, out_dir, run_id):
    os.makedirs(out_dir, exist_ok=True)

    y_pred = pred[:, 1].astype(int)
    y_pred = preprocessing.LABEL_ENCODER.inverse_transform(y_pred)
    pd.DataFrame.from_dict({'id': pred[:, 0], 'y_pred': y_pred, 'y_pred_proba': pred[:, 2]}) \
        .to_csv(join(out_dir, '{}_test.csv'.format(run_id)), index=False)


def save_cv_results(cv_results, out_dir, run_id):
    pd.DataFrame(cv_results) \
        .to_csv(join(out_dir, '{}_cv_results.csv'.format(run_id)), index=False)


def save_best_params(best_params, out_dir, run_id):
    with open(join(out_dir, '{}_best_params.csv'.format(run_id)), 'w+') as file:
        json.dump(best_params, file)


def run_exists(out_dir, run_id):
    return exists(join(out_dir, '{}_test.csv'.format(run_id)))


def main(args):
    in_dir = join(dirname(__file__), '../../data/labeled/')
    out_dir = join(dirname(__file__), '../../output/predictions/{}'.format(args.community))

    if not run_exists(out_dir, args.run_id):
        logger.info('Run %s does not exists. Start new...', args.run_id)
        pred, cv_results, best_params = make_predictions(args)
        save_predictions(pred, out_dir, args.run_id)
        save_cv_results(cv_results, out_dir, args.run_id)
        save_best_params(best_params, out_dir, args.run_id)
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
    parser.add_argument("--n_jobs", help="Parallel jobs", type=int, default=1)
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
