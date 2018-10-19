"""
Export a given Stack Exchange community to CSV file and create train/test splits.

The CSV file has following fields:
    ['Id', 'Title', 'Body', 'Tags', 'Label']

Where title, body and tag correspond to the initial text when the question has been posted.

Random seed is configured such that train/test splits are always identical.
"""
import argparse
import csv
import logging
import logging.config
import os

import numpy as np
import pandas as pd
import pymongo
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from qac.storage import post_storage

logger = logging.getLogger(__name__)  # pylint: disable=locally-disabled, invalid-name

QUERY = {
    "$or": [
        {"qac_annotation.label": {"$eq": "CLEAR"}},
        {
            "qac_annotation.label": {"$eq": "UNCLEAR"},
            "$or": [
                {"qac_annotation.clarification_edit": {"$exists": True}},
                {"qac_annotation.clarification_comment": {"$exists": True}}
            ]
        }
    ]
}


def _initial_fields(edits):
    title, body, tags = None, None, None

    for edit in edits:
        if edit['PostHistoryTypeId'] == "1":
            title = edit.get('Text', '')
        elif edit['PostHistoryTypeId'] == "2":
            body = edit.get('Text', '')
        elif edit['PostHistoryTypeId'] == "3":
            tags = edit.get('Text', '')

    return title, body, tags


def main(args):
    logger.info('Start CSV dump of %s', args.community)

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    storage = post_storage.PostStorage(community=args.community)

    instances = []

    with open(os.path.join(out_dir, '{}.csv'.format(args.community)), 'w+') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['id', 'title', 'body', 'tags', 'label'])

        count = storage.count()
        # sort query results on Id to avoid any side effects due to changing insertion order
        cursor = storage.find(QUERY).sort([('Id', pymongo.ASCENDING)])

        for post in tqdm(cursor, total=count):
            title, body, tags = _initial_fields(post['Edits'])

            if title is None or body is None:
                logger.info('Skip post %s due to insufficient fields.', post['Id'])
                continue

            instances.append([post['Id'], post['qac_annotation']['label']])
            writer.writerow([post['Id'], title, body, tags, post['qac_annotation']['label']])

    X, y = np.array(instances).T
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )
    train = np.stack([X_train, y_train], axis=1)
    test = np.stack([X_test, y_test], axis=1)

    pd.DataFrame(train, columns=['id', 'label']) \
        .to_csv(os.path.join(out_dir, '{}_train.csv'.format(args.community)), index=False)
    pd.DataFrame(test, columns=['id', 'label']) \
        .to_csv(os.path.join(out_dir, '{}_test.csv'.format(args.community)), index=False)

    logger.info('Finish CSV dump %s', args.community)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("community", help="Community name", type=str)
    parser.add_argument("output_dir", help="Output directory", type=str)
    parser.add_argument("-t", "--test_size", help="Test set size", type=float, default=0.2)
    return parser.parse_args()


if __name__ == '__main__':
    logging.config.fileConfig(os.path.join(os.path.dirname(__file__), '../logging_config.ini'),
                              disable_existing_loggers=False)
    main(arg_parser())
