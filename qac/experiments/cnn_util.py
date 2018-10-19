import csv
import inspect
import logging
import os
from contextlib import redirect_stdout
from os.path import join

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import keras
from qac.evaluation import evaluation
from qac.experiments import preprocessing

logger = logging.getLogger(__name__)  # pylint: disable=locally-disabled, invalid-name


class CNNStaticGenerator(keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size, embedding_weights, return_y=True):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.return_y = return_y
        self.embedding_weights = embedding_weights

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        embedded_x = np.stack([np.stack([self.embedding_weights[word]
                                         for word in sentence]) for sentence in batch_x])

        if self.return_y:
            return embedded_x, np.array(batch_y)

        return embedded_x


def filter_func_args(params, func):
    signature = inspect.signature(func)
    keys = list(signature.parameters.keys())
    filtered = {key: params[key] for key in params.keys() if key in keys}
    return filtered


def save_config(run_dir, config, filename='params'):
    keys = sorted(config.keys())
    with open(join(run_dir, '{}.csv'.format(filename)), 'w') as file:
        writer = csv.writer(file)
        writer.writerow(keys)
        writer.writerow([config[key] for key in keys])


def save_summary(run_dir, model):
    with open(join(run_dir, 'modelsummary.txt'), 'w') as file:
        with redirect_stdout(file):
            model.summary()


def save_history(run_dir, history):
    keys = ['epoch', 'loss', 'acc', 'val_loss', 'val_acc']
    history['epoch'] = list(range(1, len(history['loss']) + 1))

    with open(join(run_dir, 'history.csv'), 'w') as file:
        writer = csv.writer(file)
        writer.writerow(keys)
        writer.writerows(zip(*[history[key] for key in keys]))


def save_best_epoch(run_dir, early_stopping):
    stopped = early_stopping.stopped_epoch
    best = stopped - early_stopping.patience + 1
    logger.info('Best epoch %d', best)
    with open(join(run_dir, 'best_epoch.txt'), 'w') as file:
        file.write('{}\n'.format(best))


def plot_runs(out_dir, run_id, histories, title, config_names):
    f, axs = plt.subplots(nrows=2, ncols=2, figsize=(13, 10), sharex=True)

    for history, name in zip(histories, config_names):
        axs[0][0].plot(history['epoch'], history['acc'], label=name)
        axs[0][0].set_title('Accuracy')
        axs[0][1].plot(history['epoch'], history['loss'], label="_nolegend_")
        axs[0][1].set_title('Loss')
        axs[1][0].plot(history['epoch'], history['val_acc'], label="_nolegend_")
        axs[1][0].set_title('Val. Accuracy')
        axs[1][1].plot(history['epoch'], history['val_loss'], label="_nolegend_")
        axs[1][1].set_title('Val. Loss')

    f.legend(loc='upper right')
    f.suptitle(title, fontsize=14)
    f.tight_layout()
    f.subplots_adjust(top=0.93)
    f.savefig('{}/{}.png'.format(out_dir, run_id))


def consolidate_runs(args, metadata_dir, run_names, execution_id):
    run_dfs = []
    accuracies = []

    for run_name in run_names:
        run_dir = join(metadata_dir, run_name)
        val_results = pd.read_csv(join(run_dir, 'val_results.csv'))
        accuracies.append(val_results['accuracy'].values[0])
        params = pd.read_csv(join(run_dir, 'params.csv'))
        val_results['best_epoch'] = open(join(run_dir, 'best_epoch.txt')).readlines()[0].strip()
        joined = pd.concat([val_results, params], axis=1)
        joined['run_name'] = '{}-{}'.format(execution_id, run_name)
        joined.set_index('run_name', inplace=True)
        run_dfs.append(joined)

    histories = []
    n_best = np.argsort(accuracies)[-5:]
    best_runs = [run_names[i] for i in n_best]
    for run_name in best_runs:
        run_dir = join(metadata_dir, run_name)
        histories.append(pd.read_csv(join(run_dir, 'history.csv')))
    plot_runs(metadata_dir, 'training', histories, args.community, best_runs)

    pd.concat(run_dfs).to_excel(join(metadata_dir, 'summary.xlsx'))


def save_predictions(pred, out_dir, run_name):
    os.makedirs(out_dir, exist_ok=True)

    y_pred = pred[:, 1].astype(int)
    y_pred = preprocessing.LABEL_ENCODER.inverse_transform(y_pred)
    pd.DataFrame.from_dict({'id': pred[:, 0], 'y_pred': y_pred, 'y_pred_proba': pred[:, 2]}) \
        .to_csv(join(out_dir, '{}_test.csv'.format(run_name)), index=False)
