"""Stack Exchange question importer script.

Questions of a single community are imported into MongoDB alongside with their comments and edits.
"""
import argparse
import logging
import logging.config
from os.path import dirname, join

from tqdm import tqdm

from qac.dataset.parser import XMLParser
from qac.storage.post_storage import PostStorage

logger = logging.getLogger(__name__)  # pylint: disable=locally-disabled, invalid-name


class StackExchangeImporter(object):

    def __init__(self, community, path):
        """
        :param community: The SE community for which questions are imported (used as identifier).
        :param path: path to dump of respective community.
        """
        self.path = path
        self.posts = XMLParser(join(path, 'Posts.xml'))
        self.comments = XMLParser(join(path, 'Comments.xml'))
        self.edits = XMLParser(join(path, 'PostHistory.xml'))

        self.storage = PostStorage(community)

    def load(self):
        question_ids = set()
        # See https://meta.stackexchange.com/a/2678 for different edit types.
        allowed_edits = list(range(1, 10))

        logger.info('Start processing posts')
        for post in tqdm(self.posts):
            if post.PostTypeId == '1':  # questions only
                question_ids.add(post.Id)
                self.storage.save_post(post)
        logger.info('Imported %d questions', len(question_ids))

        self.storage.create_id_index()

        logger.info('Start processing comments')
        for comment in tqdm(self.comments):
            if comment.PostId in question_ids:
                self.storage.add_comment(comment.PostId, comment)

        logger.info('Start processing edits')
        for edit in tqdm(self.edits):
            if edit.PostId in question_ids and int(edit.PostHistoryTypeId) in allowed_edits:
                self.storage.add_edit(edit.PostId, edit)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("community", help="community name", type=str)
    parser.add_argument("path",
                        help="base path to data files (i.e. directory where Posts.xml, Comments.xml, PostHistory.xml are located)",
                        type=str)
    return parser.parse_args()


def main(args):
    logger.info('Start importing %s from %s', args.community, args.path)
    importer = StackExchangeImporter(args.community, args.path)
    importer.load()


if __name__ == '__main__':
    logging.config.fileConfig(join(dirname(__file__), '../logging_config.ini'),
                              disable_existing_loggers=False)
    main(arg_parser())
