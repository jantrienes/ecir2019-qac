import argparse
import glob
import logging
import logging.config
from os.path import dirname, join

from qac.evaluation import evaluation
from qac.experiments import preprocessing

logger = logging.getLogger(__name__)  # pylint: disable=locally-disabled, invalid-name


def _filter_files(files):
    return [file for file in files if file.endswith('_test.csv')]


def main(args):
    ground_truth = join(dirname(__file__), '../../data/labeled/{}_test.csv'.format(args.community))
    out_dir = join(dirname(__file__), '../../output/predictions/{}'.format(args.community))

    files = glob.glob(join(out_dir, '*.csv'))
    files = _filter_files(files)

    results_test = evaluation.evaluate(
        csv_true=ground_truth,
        csv_pred=files,
        label_encoder=preprocessing.LABEL_ENCODER
    )
    report_dir = join(dirname(__file__), '../../output/evaluation/')
    evaluation.save_results_csv(results_test, report_dir, args.community)
    evaluation.save_results_excel(results_test, report_dir, args.community)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("community", help="Community name", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    logging.config.fileConfig(join(dirname(__file__), '../logging_config.ini'),
                              disable_existing_loggers=False)
    main(arg_parser())
