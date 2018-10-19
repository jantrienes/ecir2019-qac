"""
Export clarification questions for a given Stack Exchange community to CSV file.

The CSV file has following fields:
    ['id', 'clarification_question']

Clarification questions are only present for unclear questions of that given community.
"""
import argparse
import csv
import logging
import logging.config
import os

from tqdm import tqdm

from qac.storage import post_storage

logger = logging.getLogger(__name__)  # pylint: disable=locally-disabled, invalid-name

QUERY = {
    "qac_annotation.label": {"$eq": "UNCLEAR"},
    "$or": [
        {"qac_annotation.clarification_edit": {"$exists": True}},
        {"qac_annotation.clarification_comment": {"$exists": True}}
    ]
}


def main(args):
    logger.info('Start CSV dump of %s', args.community)

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    storage = post_storage.PostStorage(community=args.community)

    with open(os.path.join(out_dir, '{}.csv'.format(args.community)), 'w+') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['id', 'clarification_question'])

        # sort query results on Id to avoid any side effects due to changing insertion order
        cursor = storage.find(QUERY)

        for post in tqdm(cursor):
            row = [
                post['Id'],
                post['qac_annotation']['question']['Text'],
            ]
            writer.writerow(row)

    logger.info('Finish CSV dump %s', args.community)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("community", help="Community name", type=str)
    parser.add_argument("output_dir", help="Output directory", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    logging.config.fileConfig(os.path.join(os.path.dirname(__file__), '../logging_config.ini'),
                              disable_existing_loggers=False)
    main(arg_parser())
