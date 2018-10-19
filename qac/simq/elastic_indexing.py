import logging
import os

import elasticsearch
import elasticsearch.exceptions
import elasticsearch.helpers

from qac.experiments import preprocessing

logger = logging.getLogger(__name__)  # pylint: disable=locally-disabled, invalid-name

INDEX_SETTINGS = {
    "settings": {
        "similarity": {
            "my_bm25": {
                "type": "BM25",
                "b":    0.75,
                "k1": 1.2
            }
        }
    },
    "mappings": {
        "post": {
            "_all": {
                "enabled": False
            },
            "properties": {
                "title": {
                    "type": "string",
                    "analyzer": "stop",
                    "similarity": "my_bm25"
                },
                "body": {
                    "type": "string",
                    "analyzer": "stop",
                    "similarity": "my_bm25"
                },
                "tags": {
                    "type": "string",
                    "analyzer": "stop",
                    "similarity": "my_bm25"
                },
                "combined": {
                    "type": "string",
                    "analyzer": "stop",
                    "similarity": "my_bm25"
                },
                "label": {
                    "type": "string",
                    "analyzer": "whitespace",
                }
            }
        }
    }
}


def connect_es():
    es_host = os.getenv('ELASTIC_HOST', 'localhost')
    es_port = os.getenv('ELASTIC_PORT', '9200')
    es = elasticsearch.Elasticsearch(['http://{}:{}'.format(es_host, es_port)])
    return es


def create_index(es, index_name, body):
    es.indices.delete(index=index_name, ignore=[400, 404])
    es.indices.create(index=index_name, body=body)


def index_documents(es, actions, index_name, body):
    create_index(es, index_name, body)

    return elasticsearch.helpers.bulk(
        es,
        actions,
        chunk_size=40000,
        request_timeout=30,
        refresh='wait_for'
    )


def get_post_document(post, post_id, label):
    title, body, tags = post[0], post[1], post[2]

    title = preprocessing.get_tokens(title)
    body = preprocessing.get_tokens(body)
    tags = preprocessing.get_tokens(tags)

    return {
        '_id': str(post_id),
        'title': ' '.join(title),
        'body': ' '.join(body),
        'tags': ' '.join(tags),
        'combined': ' '.join(title + body + tags),
        'label': label
    }


def get_actions(index_name, doc_type, posts, labels, post_ids):
    for post, post_id, label in zip(posts, post_ids, labels):
        yield {
            '_op_type': 'index',
            '_index': index_name,
            '_type': doc_type,
            **get_post_document(post, post_id, label)
        }


def query(es, index, doc_type, field, text, N=10):
    q = {
        'size': N,
        'query': {
            'match': {
                field: text
            }
        }
    }

    return es.search(index, doc_type, q, request_timeout=10)


def index_posts(es, index_name, doc_type, X, y, ids, body=INDEX_SETTINGS):
    index = True

    try:
        count = es.count(index_name)['count']
        if count == len(ids):
            index = False
    except elasticsearch.exceptions.NotFoundError:
        index = True

    if index:
        logger.info('Start indexing documents into "%s"...', index_name)
        actions = get_actions(index_name, doc_type, X, y, ids)
        index_documents(es, actions, index_name, body)
    else:
        logger.info('Index "%s" already exists. Skip indexing.', index_name)
