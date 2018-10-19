import argparse
import logging
import logging.config
import os
from os.path import dirname, exists, join

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler

from qac.simq import simq_features
from qac.evaluation import evaluation
from qac.experiments import preprocessing

logger = logging.getLogger(__name__)  # pylint: disable=locally-disabled, invalid-name

DEBUG = False
THRESHOLDS = np.linspace(0, 1, num=1000)


class ThresholdClassifier():

    def __init__(self, T):
        self.T = T

    def predict(self, X):
        return np.where(X >= self.T, 0, 1)

    def decision_function(self, X):
        y = self.predict(X)
        return np.where(y == 0, self.T - X, X - self.T)


def _scaled_scores(scores, feat, train_ids, val_ids, test_ids):
    X_train = scores[feat].loc[train_ids].values.reshape(-1, 1)
    X_val = scores[feat].loc[val_ids].values.reshape(-1, 1)
    X_test = scores[feat].loc[test_ids].values.reshape(-1, 1)
    scaler = StandardScaler().fit(X_train)
    X_val = scaler.transform(X_val)[:, 0]
    X_test = scaler.transform(X_test)[:, 0]
    return X_val, X_test


def make_predictions(args):
    community = preprocessing.load_community(args.community, preprocess=False, with_dev=True)
    train_ids, val_ids, test_ids = community.train_ids, community.val_ids, community.test_ids
    y_val = community.y_val

    feature_handler = simq_features.FeatureHandler(args.community, run=args.simq_run)
    feat = args.feature
    scores = feature_handler.read(feature_name=feat)

    X_val, X_test = _scaled_scores(scores, args.feature, train_ids, val_ids, test_ids)

    best_t = evaluate_thresholds(THRESHOLDS, X_val, y_val)
    clf = ThresholdClassifier(best_t)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.decision_function(X_test)

    return np.stack([test_ids, y_pred, y_pred_proba], axis=1)


def evaluate_thresholds(thresholds, X_val, y_val):
    logger.info('Start threshold search...')
    scores = []
    for T in thresholds:
        clf = ThresholdClassifier(T)

        y_pred = clf.predict(X_val)
        y_pred_proba = clf.decision_function(X_val)

        acc_score = accuracy_score(y_val, y_pred)
        auc_score = roc_auc_score(y_val, y_pred_proba)
        f1 = f1_score(y_val, y_pred)

        if DEBUG:
            logger.info('Acc - %.4f AUC - %.4f F1 - %.4f', acc_score, auc_score, f1)

        scores.append(auc_score)

    i = np.argmax(scores)
    logger.info('Best threshold = %s', str(thresholds[i]))
    logger.info('Best score = %.4f', scores[i])
    return thresholds[i]


def save_predictions(pred, out_dir, run_id):
    os.makedirs(out_dir, exist_ok=True)

    y_pred = pred[:, 1].astype(int)
    y_pred = preprocessing.LABEL_ENCODER.inverse_transform(y_pred)
    pd.DataFrame.from_dict({'id': pred[:, 0], 'y_pred': y_pred, 'y_pred_proba': pred[:, 2]}) \
        .to_csv(join(out_dir, '{}_test.csv'.format(run_id)), index=False)


def run_exists(out_dir, run_id):
    return exists(join(out_dir, '{}_test.csv'.format(run_id)))


def _run_id(args):
    return 'simq_{}_threshold_{}'.format(args.simq_run, args.feature)


def main(args):
    in_dir = join(dirname(__file__), '../../data/labeled/')
    out_dir = join(dirname(__file__), '../../output/predictions/{}'.format(args.community))

    run_id = _run_id(args)
    if not run_exists(out_dir, run_id) or DEBUG:
        logger.info('Run %s does not exists. Start new...', run_id)
        pred = make_predictions(args)
        save_predictions(pred, out_dir, run_id)
        logger.info('Finish training and testing.')

    results_test = evaluation.evaluate(
        csv_true=join(in_dir, '{}_test.csv'.format(args.community)),
        csv_pred=join(out_dir, '{}_test.csv'.format(run_id)),
        label_encoder=preprocessing.LABEL_ENCODER
    )
    evaluation.print_results(results_test, '{}_test'.format(run_id))


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("community", help="Community name", type=str)
    parser.add_argument("simq_run", help="Identifier of similar question retrieval run", type=str)
    parser.add_argument("feature", help="Identifier of feature", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()
    log_dir = join(dirname(__file__), '../../output/predictions/{}'.format(args.community))
    os.makedirs(log_dir, exist_ok=True)
    log_filename = join(log_dir, '{}.log'.format(_run_id(args)))
    logging.getLogger('elasticsearch').setLevel(logging.WARNING)
    logging.config.fileConfig(join(dirname(__file__), '../logging_file.ini'),
                              disable_existing_loggers=False,
                              defaults={'logfilename': log_filename})
    main(args)
