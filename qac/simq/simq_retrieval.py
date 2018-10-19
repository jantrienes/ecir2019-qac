import argparse
import logging
import logging.config
import os
from itertools import chain
from os.path import dirname, join, exists

import numpy as np
from tqdm import tqdm

from qac.simq import elastic_indexing
from qac.experiments import preprocessing

N_SIM = 60

logger = logging.getLogger(__name__)  # pylint: disable=locally-disabled, invalid-name


def index_unclear(es, index_name, X, y, ids):
    unclear_ixs = np.where(y == 1)[0]
    X = np.take(X, unclear_ixs, axis=0)
    y = np.take(y.values, unclear_ixs, axis=0).tolist()
    ids = np.take(ids, unclear_ixs, axis=0).tolist()

    elastic_indexing.index_posts(es, index_name, 'post', X, y, ids)


def query_text(args, post):
    title, body, tags = post[0], post[1], post[2]
    title = preprocessing.get_tokens(title)
    tags = preprocessing.get_tokens(tags)

    query = ''
    if args.strategy == 'constrained':
        query = ' '.join(title + tags)
    elif args.strategy == 'all':
        body = preprocessing.get_tokens(body)
        # remove excessively long tokens (e.g. stack dumps on unix)
        body = [token for token in body if len(token) <= 40]
        body = body[:args.body_length]
        query = ' '.join(title + body + tags)
    else:
        raise ValueError('Unknown query strategy "{}"'.format(args.strategy))

    return query



def make_run(args, es, index_name, run_name, X, ids, total):
    simq_path = join(dirname(__file__), '../../models/simq/')
    simq_file = join(simq_path, '{}_{}.run'.format(index_name, run_name))
    os.makedirs(simq_path, exist_ok=True)

    with open(simq_file, 'w') as run_file:
        for post, post_id in tqdm(zip(X, ids), total=total):
            query = query_text(args, post)

            try:
                result = elastic_indexing.query(es, index_name, 'post', 'combined', query, N=N_SIM)
                hits = result['hits']['hits']
                for i, hit in enumerate(hits):
                    run_file.write('{} Q0 {} {} {} {}\n'.format(
                        post_id, hit['_id'], i, hit['_score'], run_name))
            except Exception as e:
                logger.warning('Exception during querying of post "%s"', post_id)
                logger.warning(e)


def main(args):
    X_train, y_train, train_ids, X_val, _, val_ids, X_test, _, test_ids = preprocessing \
        .load_community(args.community, preprocess=False, with_dev=True)

    es = elastic_indexing.connect_es()
    index_name_all = 'stackexchange_{}'.format(args.community)
    index_name_unclear = '{}_unclear'.format(index_name_all)

    elastic_indexing.index_posts(es, index_name_all, 'post', X_train, y_train, train_ids)
    index_unclear(es, index_name_unclear, X_train, y_train, train_ids)

    if args.debug:
        for post, post_id in zip(X_train[:5], train_ids[:5]):
            print('Id = {}'.format(post_id))
            print('Title = {}'.format(post[0]))
            print('Tags = {}'.format(post[2]))
            query = query_text(args, post)
            print('Query = {}'.format(query))
            result = elastic_indexing.query(es, index_name_all, 'post', 'combined', query, N=N_SIM)

            hits = result['hits']['hits'][:10]
            for hit in hits:
                print('{} {:.4f} {:}'.format(hit['_id'], hit['_score'], hit['_source']['title']))

            print('========================')
        return

    total = len(train_ids) + len(val_ids) + len(test_ids)
    def posts(): return chain(X_train, X_val, X_test)
    def ids(): return chain(train_ids, val_ids, test_ids)

    logger.info('Start to query "all" index for similar posts...')
    make_run(args, es, index_name_all, args.run_id, posts(), ids(), total)
    logger.info('Start to query "unclear" index for similar posts...')
    make_run(args, es, index_name_unclear, args.run_id, posts(), ids(), total)

def run_exists(args):
    simq_path = join(dirname(__file__), '../../models/simq/')
    run_all = 'stackexchange_{}_{}.run'.format(args.community, args.run_id)
    run_unclear = 'stackexchange_{}_unclear_{}.run'.format(args.community, args.run_id)
    simq_file = join(simq_path, run_all)
    simq_file_unclear = join(simq_path, run_unclear)
    return exists(simq_file) and exists(simq_file_unclear)

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("community", help="Community name", type=str)
    parser.add_argument("run_id", help="Run identifier", type=str)
    parser.add_argument("--debug", help="Debug mode", type=bool, default=False)
    parser.add_argument("--body_length",
                        help="When strategy=all, constrain question body to this many tokens.",
                        type=int,
                        default=100)
    parser.add_argument("--strategy",
                        help="Query strategy",
                        type=str,
                        choices=['all', 'constrained'],
                        default='all')
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parser()
    log_dir = join(dirname(__file__), '../../models/simq/')
    os.makedirs(log_dir, exist_ok=True)
    log_filename = join(log_dir, 'stackexchange_{}_{}.log'.format(args.community, args.run_id))
    logging.config.fileConfig(join(dirname(__file__), '../logging_file.ini'),
                              disable_existing_loggers=False,
                              defaults={'logfilename': log_filename})
    logging.getLogger('elasticsearch').setLevel(logging.WARNING)

    if run_exists(args):
        logger.info('Retrieval results for for community "%s" with ID "%s" already exist....',
                    args.community, args.run_id)
    else:
        main(args)
