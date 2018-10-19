import logging
import os

import pymongo


logger = logging.getLogger(__name__)  # pylint: disable=locally-disabled, invalid-name


class Storage(object):

    def __init__(self, host=None, port=None, db=None):
        if host is None:
            host = os.getenv('MONGO_HOST', 'localhost')

        if port is None:
            port = int(os.getenv('MONGO_PORT', '27017'))

        if db is None:
            db = os.getenv('MONGO_DB', 'stackexchange')

        mongo_uri = os.getenv('MONGO_URI', False)
        if mongo_uri:
            logger.debug('MONGO_URI=%s', mongo_uri)
            self._client = pymongo.MongoClient(mongo_uri)
            try:
                self._db = self._client.get_database()
            except TypeError:
                db_name = mongo_uri.split('/')[-1]
                logger.info('PyMongo v=%s, manually extracted DB name "%s" from URI.',
                            pymongo.__version__, db_name)
                self._db = self._client.get_database(name=db_name)
        else:
            logger.debug('host=%s', host)
            logger.debug('port=%s', port)
            self._client = pymongo.MongoClient(host, port)
            self._db = self._client[db]


    def communities(self):
        result = []
        for c in self._db.collection_names():
            if 'posts' in c:
                result.append('_'.join(c.split('_')[1:]))

        return result

    def collections(self):
        return self._db.collection_names()

    def create_index(self, collection, fields):
        indexes = []
        for field in fields:
            indexes.append((field, pymongo.ASCENDING))

        return self._db[collection].create_index(indexes)


    def users(self):
        return self._db['users'].find({})

    def add_user(self, name):
        return self._db['users'].insert_one({'name': name})
