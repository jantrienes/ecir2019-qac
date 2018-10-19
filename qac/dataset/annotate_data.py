import argparse
import logging
import logging.config
import os

from tqdm import tqdm

from qac.dataset import annotation
from qac.dataset.util import print_statistics
from qac.storage import post_storage

logger = logging.getLogger(__name__)  # pylint: disable=locally-disabled, invalid-name


class StackExchangeAnnotator(object):

    def __init__(self, community):
        self.storage = post_storage.PostStorage(community)
        self.annotator = annotation.Annotator()

    def _annotate(self, post):
        label, question, comment, edit = None, None, None, None

        label = self.annotator.assign_label(post)
        if label == annotation.QuestionLabels.UNCLEAR:
            question = self.annotator.clarification_question(post)
            if question:
                comment, edit = self.annotator.clarification(post, question)

        self.storage.add_annotation(post['Id'], label, question, comment, edit)

    def start_annotation(self):
        total = self.storage.count()

        for post in tqdm(self.storage.get_all(), total=total):
            try:
                self._annotate(post)
            except:
                logger.warning('Failed to annotate post (Id=%s)', str(post['Id']), exc_info=1)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("community", help="community name", type=str)
    return parser.parse_args()


def main(args):
    logger.info('Start annotating %s', args.community)
    annotator = StackExchangeAnnotator(args.community)
    annotator.start_annotation()
    logger.info('Finish annotating %s', args.community)
    print_statistics(annotator.storage.annotation_statistics())


if __name__ == '__main__':
    logging.config.fileConfig(os.path.join(os.path.dirname(__file__), '../logging_config.ini'),
                              disable_existing_loggers=False)
    main(arg_parser())
