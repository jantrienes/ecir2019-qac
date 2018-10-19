import argparse
import logging
import logging.config
import os
from collections import Counter, defaultdict
from os.path import dirname, join

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from qac.experiments import preprocessing

logger = logging.getLogger(__name__)  # pylint: disable=locally-disabled, invalid-name
DF_COUNTS_THRESHOLD = 10


class SummaryStatistics(object):

    def __init__(self, community, csv_file):
        self.community = community
        self.csv_file = csv_file

        base_path = dirname(__file__)
        results_path = '../../output/reports/{}'.format(community)
        self.output_dir = join(base_path, results_path)
        os.makedirs(self.output_dir, exist_ok=True)

    def document_statistics(self, tags, title, body):
        text = preprocessing.join_question_fields(tags, title, body)
        tokens = preprocessing.get_tokens(text)
        tokens_filtered = [t for t in tokens if t.isalpha()]
        return len(tokens), Counter(tokens), set(tokens_filtered)

    def generate_report(self):
        df = preprocessing.load(self.csv_file)

        samples = len(df)
        classes = len(df.label.unique())
        samples_by_class = df.label.value_counts()

        logger.info('Estimating vocabulary statistics.')
        corpus_token_dist = Counter()
        document_frequency = Counter()
        cleaned = set()
        lengths = []
        for row in tqdm(df.itertuples(), total=len(df)):
            length, doc_dist, tokens_filtered = self.document_statistics(
                getattr(row, 'tags'), getattr(row, 'title'), getattr(row, 'body'))
            lengths.append(length)
            document_frequency.update(doc_dist.keys())
            corpus_token_dist.update(doc_dist)
            cleaned.update(tokens_filtered)
        logger.info('Reporting statistics')

        df['length'] = lengths
        median_length = np.median(lengths)
        min_length = np.min(lengths)
        max_length = np.max(lengths)

        voc_size = len(corpus_token_dist)
        voc_size_filtered = len(cleaned)

        document_frequency = self._threshold_counts(document_frequency, DF_COUNTS_THRESHOLD)
        self._plot_df_dist(document_frequency, DF_COUNTS_THRESHOLD)

        with open(join(self.output_dir, 'statistics.txt'), 'w+') as file:
            file.write('Samples = {}\n'.format(samples))
            file.write('Classes = {}\n'.format(classes))
            file.write('Distribution\n')
            for label, count in samples_by_class.iteritems():
                file.write('\t{0: <9} {1:} ({2:.2f})\n'.format(label, count, count / samples))
            file.write('Median Sample Length = {}\n'.format(median_length))
            file.write('Min Sample Length = {}\n'.format(min_length))
            file.write('Max Sample Length = {}\n'.format(max_length))
            file.write('Vocabulary Size = {}\n'.format(voc_size))
            file.write('Vocabulary Size (min_df=3) = {}\n'.format(document_frequency.loc[3:].sum()))
            file.write('Vocabulary Size Filtered (only alpha) = {}\n\n'.format(voc_size_filtered))
            file.write('Most Common Words:\n{}\n\n'.format(corpus_token_dist.most_common()[:50]))
            file.write('Least Common Words:\n{}\n\n'.format(corpus_token_dist.most_common()[-10:]))
            file.write('Document Frequency distribution:\n{}'.format(document_frequency))

        self._plot_length_distribution(lengths, median_length, self.community, 'ALL')
        for label in df.label.unique():
            lengths = df[df.label == label].length.tolist()
            self._plot_length_distribution(lengths, np.median(lengths), self.community, label)

    def _plot_length_distribution(self, lengths, median, community, label):
        filename = join(self.output_dir, 'distribution_sample_length_{}.pdf'.format(label))
        with PdfPages(filename) as pdf_fig:
            fig = plt.figure(figsize=(16, 9))
            plt.hist(lengths, bins=500)

            plt.axvline(median, c='r', label='Median ({})'.format(int(median)), lw=.8)
            plt.title('Sample Length Distribution ({} {})'.format(community, label))
            plt.xlabel('Length (Tokens)')
            plt.ylabel('Frequency')
            plt.legend()
            plt.xlim(0, 3000)
            pdf_fig.savefig(fig, bbox_inches='tight')

    def _plot_df_dist(self, df, threshold):
        relative = df / df.sum()
        relative = relative.sort_index()
        filename = join(self.output_dir, 'distribution_df.pdf')
        with PdfPages(filename) as pdf_fig:
            ax = relative.plot.bar(color='lightblue', figsize=(11, 7))

            for i in ax.patches:
                ax.text(i.get_x() + .07, i.get_height() + 0.007,
                        '{:d}%'.format(int(i.get_height() * 100)), fontsize=10,
                        color='dimgrey')

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.title('Document Frequency Distribution {} (n={})'.format(self.community, df.sum()))
            plt.xlabel('Documents')
            ticks = list(map(str, range(1, threshold + 1))) + ['>{}'.format(threshold + 1)]
            ax.set_xticklabels(ticks, rotation=0)
            plt.yticks([])
            pdf_fig.savefig(ax.get_figure(), bbox_inches='tight')

    def _threshold_counts(self, counter, threshold):
        counts = defaultdict(int)

        for _, count in counter.items():
            if count <= threshold:
                counts[count] += 1
            else:
                counts[threshold + 1] += 1

        return pd.Series(counts)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("community", help="community name", type=str)
    parser.add_argument("csv_file", help="csv file", type=str)
    return parser.parse_args()


def main(args):
    logger.info('Generate statistics for %s', args.community)
    stats = SummaryStatistics(args.community, args.csv_file)
    stats.generate_report()
    logger.info('Finish generating statistics for %s', args.community)


if __name__ == '__main__':
    logging.config.fileConfig(join(dirname(__file__), '../logging_config.ini'),
                              disable_existing_loggers=False)
    main(arg_parser())
