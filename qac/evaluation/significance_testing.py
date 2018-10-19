import argparse
import csv
import logging
import logging.config
from collections import namedtuple
from os.path import dirname, join

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

from qac.experiments import preprocessing
from qac.evaluation.art import ApproximateRandomizationTest

logger = logging.getLogger(__name__)  # pylint: disable=locally-disabled, invalid-name

Run = namedtuple('Run', ['name', 'predictions'])


class SignificanceReport():

    def __init__(self, community, metrics, n_jobs):
        self.community = community
        self.metrics = metrics
        self.n_jobs = n_jobs
        self.ground_truth = preprocessing.LABEL_ENCODER.transform(self.load_truth())
        self.out_file = join(
            dirname(__file__),
            '../../output/evaluation/significance_{}.csv'.format(self.community))

    def run_path(self, run_name):
        return join(dirname(__file__), '../../output/predictions/', self.community, run_name)

    def load_run(self, run_name):
        y_pred = pd.read_csv(self.run_path(run_name)).y_pred.values
        return Run(run_name, preprocessing.LABEL_ENCODER.transform(y_pred))

    def load_truth(self):
        filename = join(dirname(__file__), '../../data/labeled/',
                        '{}_test.csv'.format(self.community))
        y_true = pd.read_csv(filename).label.values
        return y_true

    def art_test(self, run_a, run_b, metric):
        art = ApproximateRandomizationTest(
            self.ground_truth, run_a, run_b, metric, trials=1000, n_jobs=self.n_jobs)
        return art.run()

    def make_report(self):
        logger.info('Load prediction outcomes...')
        rand = self.load_run('baseline_random_test.csv')
        majority = self.load_run('baseline_majority_test.csv')
        bow_lr_n1 = self.load_run('baseline_lr_1ngram_c1_test.csv')
        bow_lr_n3 = self.load_run('baseline_lr_3ngram_c1_test.csv')
        cnn = self.load_run('baseline_cnn_test.csv')
        simq_majority = self.load_run('simq_60stop0body_majority_test.csv')
        cq_global = self.load_run('simq_60stop0body_threshold_feat_unclear_global_cos_test.csv')
        cq_individual = self.load_run(
            'simq_60stop0body_threshold_feat_unclear_individual_cos_test.csv')
        cq_weighted = self.load_run(
            'simq_60stop0body_threshold_feat_unclear_individual_cos_weighted_test.csv')
        simq_ml = self.load_run('simq_60stop0body_ml_all_test.csv')

        pairs = [
            (majority, rand),
            (bow_lr_n1, majority),
            (bow_lr_n3, bow_lr_n1),
            (cnn, bow_lr_n3),
            (simq_majority, bow_lr_n3),
            (cq_global, simq_majority),
            (cq_global, bow_lr_n3),
            (cq_individual, cq_global),
            (cq_individual, bow_lr_n3),
            (cq_weighted, cq_individual),
            (cq_weighted, bow_lr_n3),
            (simq_ml, cq_weighted),
            (simq_ml, bow_lr_n3)
        ]

        logger.info('Start significance testing...')

        with open(self.out_file, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['run_a', 'run_b', 'metric', 'p_value'])
            for run_a, run_b in tqdm(pairs):
                for metric in self.metrics:
                    p_value = self.art_test(run_a.predictions, run_b.predictions, metric)
                    writer.writerow([run_a.name, run_b.name, metric.__name__, p_value])

        logger.info('Finish significance testing')


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("community", help="Community name", type=str)
    parser.add_argument("--n_jobs", help="Parallel jobs", type=int, default=1)
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()
    logging.config.fileConfig(join(dirname(__file__), '../logging_config.ini'),
                              disable_existing_loggers=False)
    report = SignificanceReport(args.community,
                                metrics=[accuracy_score, roc_auc_score, f1_score],
                                n_jobs=args.n_jobs)
    report.make_report()
