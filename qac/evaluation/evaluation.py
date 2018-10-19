import logging
import os
from os.path import abspath, basename, join

import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)

logger = logging.getLogger(__name__)  # pylint: disable=locally-disabled, invalid-name


def _load_predictions(csv_pred):
    predictions = []

    if isinstance(csv_pred, str):
        csv_pred = [csv_pred]
    for csv in csv_pred:
        y_pred = pd.read_csv(csv)['y_pred']
        y_pred_proba = pd.read_csv(csv)['y_pred_proba']
        predictions.append((basename(csv), y_pred, y_pred_proba))

    return predictions


def _evaluate(y_true, y_pred, y_pred_proba, class_names):
    scores = {}
    scores['accuracy'] = accuracy_score(y_true, y_pred)
    scores['auc'] = roc_auc_score(y_true, y_pred_proba)
    prfs = precision_recall_fscore_support(y_true, y_pred)

    for ((p, r, f, s), cls) in zip(zip(*prfs), class_names):
        scores['precision_{}'.format(cls)] = p
        scores['recall_{}'.format(cls)] = r
        scores['f1_{}'.format(cls)] = f
        scores['support_{}'.format(cls)] = int(s)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    scores['tn'] = tn
    scores['fp'] = fp
    scores['fn'] = fn
    scores['tp'] = tp

    return scores


def evaluate(csv_true, csv_pred, label_encoder):
    """Evaluate (possibly multiple) prediction outcomes.

    The CSV file is expected to have a 'label' and 'label_proba' column. The probability corresponds to the positive class.

    :param csv_true: Filename of csv with ground truth.
    :param csv_pred: Single CSV filename or list of CSV filenames
    :param label_encoder: Fitted scikit-learn label encoder.
    """
    y_true = pd.read_csv(csv_true)['label']
    y_true = label_encoder.transform(y_true)
    class_names = label_encoder.classes_
    predictions = _load_predictions(csv_pred)

    files = []
    scores = []
    # each file_id corresponds to different parameter settings. e.g., svm_c0.1_l1.csv
    for file_id, y_pred, y_pred_proba in predictions:
        files.append(file_id)
        y_pred = label_encoder.transform(y_pred)
        scores.append(_evaluate(y_true, y_pred, y_pred_proba, class_names))

    return pd.DataFrame(scores, index=files)


def save_results_csv(results, out_dir, name):
    out_dir = abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    results.to_csv(join(out_dir, '{}.csv'.format(name)))


def save_results_excel(results, out_dir, name):
    out_dir = abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    columns = ['accuracy', 'auc',
               'f1_UNCLEAR', 'precision_UNCLEAR', 'recall_UNCLEAR',
               'f1_CLEAR', 'precision_CLEAR', 'recall_CLEAR',
               'support_CLEAR', 'support_UNCLEAR',
               'tn', 'fp', 'fn', 'tp']
    results = results[columns].sort_values(by='accuracy')
    results.to_excel(join(out_dir, '{}.xlsx'.format(name)), sheet_name=name)


def print_results(results, run_id, round=0):
    metrics = ['accuracy', 'auc', 'f1_CLEAR', 'f1_UNCLEAR', 'precision_CLEAR', 'precision_UNCLEAR',
               'recall_CLEAR', 'recall_UNCLEAR', 'support_CLEAR', 'support_UNCLEAR']
    results_string = 'Results for run {}\n'.format(run_id)
    results_string += '==================================\n'
    results = results[metrics]
    if round > 0:
        results_string += results.transpose().round(round).to_string() + '\n'
    else:
        results_string += results.transpose().to_string() + '\n'
    results_string += '=================================='
    logger.info('\n' + results_string)
