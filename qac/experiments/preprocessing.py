import itertools
import json
import logging
import re
from collections import Counter, namedtuple
from os.path import dirname, exists, join, split

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

URL_REGEX = re.compile(r"(https?:\/\/[^ )]+)", re.MULTILINE)
SPLIT_TAGS = re.compile('<(.*?)>')
CODE_REGEX = re.compile(r'^\ {4,}.*', re.MULTILINE)
BLOCKQUOTE_REGEX = re.compile(r'^>', re.MULTILINE)

LABEL_ENCODER = LabelEncoder().fit(['CLEAR', 'UNCLEAR'])

logger = logging.getLogger(__name__)  # pylint: disable=locally-disabled, invalid-name


def join_question_fields(tags, title, body):
    tags_formatted = ' '.join(SPLIT_TAGS.findall(tags))
    return '{} {} {}'.format(title, body, tags_formatted)


def clean_str(string):
    """
    Adapted from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = URL_REGEX.sub("URL", string)
    string = re.sub(r"[^A-Za-z0-9\.(),!?\']", " ", string)  # remove special characters
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def get_tokens(string):
    string = clean_str(string)
    return string.split(" ")


def pad_sequence(sequence, max_length, pad_char='<PAD>'):
    seq_len = len(sequence)
    padded = sequence
    if seq_len >= max_length:
        padded = sequence[:max_length]
    else:
        padded = sequence + [pad_char] * (max_length - seq_len)

    return padded


def build_vocab(docs):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*docs))
    # Mapping from index to word
    vocabulary_inv_list = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv_list)}
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
    return vocabulary, vocabulary_inv


def map_vocabulary(docs, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    mapped = np.array([[vocabulary[word] for word in doc] for doc in docs])
    return mapped


def load(csv_file):
    df = pd.read_csv(csv_file)
    df.tags.fillna('', inplace=True)
    return df


def load_community_csv(data_csv, train_csv, test_csv):
    logger.info('Load CSV data...')
    df = load(data_csv).set_index('id')
    df.label = LABEL_ENCODER.transform(df.label)

    train_ids = pd.read_csv(train_csv)['id'].values
    train_df = df.loc[train_ids]
    X_train, y_train = train_df[['title', 'body', 'tags']].values, train_df.label

    test_ids = pd.read_csv(test_csv)['id'].values
    test_df = df.loc[test_ids]
    X_test, y_test = test_df[['title', 'body', 'tags']].values, test_df.label

    return X_train, y_train, X_test, y_test, train_ids, test_ids


def _preprocess_doc(doc):
    title, body, tags = doc[0], doc[1], doc[2]
    text = join_question_fields(tags, title, body)
    return get_tokens(text)


def _filter_doc(doc, doc_freq, min_df, oov_token):
    return [token if doc_freq[token] >= min_df else oov_token for token in doc]


def _doc_frequency(data, community):
    name = '{}_vocabulary.json'.format(community)
    path = join(dirname(__file__), '../../models', name)

    doc_freq = Counter()

    if exists(path):
        logger.info('Load existing document frequency counts')
        doc_freq = Counter(json.load(open(path)))
    else:
        logger.info('Determine document frequency counts...')
        for row in data:
            tokens = _preprocess_doc(row)
            doc_freq.update(set(tokens))
        logger.info('Save document frequency counts: %s', split(path)[-1])
        json.dump(doc_freq, open(path, mode='w'))

    return doc_freq


def _preprocess(X_train, X_test, min_df, community, oov_token):
    doc_freq = _doc_frequency(itertools.chain(X_train, X_test), community)
    voc_size = sum(1 for df in doc_freq.values() if df >= min_df)
    logger.info('Vocabulary size with min_df=%d: %d', min_df, voc_size)

    logger.info('Tokenize documents...')
    docs = []
    for row in itertools.chain(X_train, X_test):
        tokens = _preprocess_doc(row)
        tokens = _filter_doc(tokens, doc_freq, min_df, oov_token)
        docs.append(tokens)

    return docs[:len(X_train)], docs[len(X_train):]


def load_community(community, preprocess=False, min_df=1, oov_token='<UNK>', with_dev=False):
    in_dir = join(dirname(__file__), '../../data/labeled/')

    data_csv = join(in_dir, '{}.csv'.format(community))
    train_csv = join(in_dir, '{}_train.csv'.format(community))
    test_csv = join(in_dir, '{}_test.csv'.format(community))

    X_train, y_train, X_test, y_test, train_ids, test_ids = load_community_csv(
        data_csv, train_csv, test_csv
    )

    if preprocess:
        X_train, X_test = _preprocess(X_train, X_test, min_df, community, oov_token)

    if with_dev:
        community = namedtuple('community', ['X_train', 'y_train', 'train_ids', 'X_val', 'y_val',
                                             'val_ids', 'X_test', 'y_test', 'test_ids'])
        X_train, X_val, y_train, y_val, train_ids, val_ids = train_test_split(
            X_train, y_train, train_ids, test_size=0.20, random_state=42, stratify=y_train)
        return community(X_train, y_train, train_ids, X_val, y_val, val_ids, X_test, y_test,
                         test_ids)

    return X_train, y_train, X_test, y_test, train_ids, test_ids

def load_community_raw(community):
    in_dir = join(dirname(__file__), '../../data/labeled/')
    data_csv = join(in_dir, '{}.csv'.format(community))
    df = load(data_csv)
    df.label = LABEL_ENCODER.transform(df.label)
    return df

def load_clarqs(community):
    clarq_path = join(dirname(__file__), '../../data/clarq/{}.csv'.format(community))
    return pd.read_csv(clarq_path)
