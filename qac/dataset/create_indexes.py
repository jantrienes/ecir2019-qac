"""
Script to create indexes on all posts collection in the Stack Exchange database for efficient
querying on Id's and annotation labels.

Mongo connection is parameterized through the MONGO_URI environment variable.
"""
import logging
import logging.config
import os

from qac.storage import base

logger = logging.getLogger(__name__)  # pylint: disable=locally-disabled, invalid-name


def main():
    fields = ['Id', 'qac_annotation.label']

    mongo_store = base.Storage()
    for collection in mongo_store.collections():
        if 'posts' in collection:
            logger.info('Create index on collection="%s" for fields="%s"', collection, fields)
            mongo_store.create_index(collection, fields=fields)
            logger.info('Indexes %s', mongo_store._db[collection].index_information().keys())


if __name__ == '__main__':
    logging.config.fileConfig(os.path.join(os.path.dirname(__file__), '../logging_config.ini'),
                              disable_existing_loggers=False)

    main()
