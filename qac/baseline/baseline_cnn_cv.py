import argparse
import logging
import logging.config
import os
from collections import namedtuple
from datetime import datetime
from os.path import dirname, join

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

import keras
import tensorflow
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import imdb
from keras.layers import (Convolution1D, Dense, Dropout, Embedding, Flatten,
                          Input, MaxPooling1D)
from keras.layers.merge import Concatenate
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.preprocessing import sequence
from qac.evaluation import evaluation
from qac.experiments import cnn_util, preprocessing, w2v
from qac.experiments.cnn_util import CNNStaticGenerator

np.random.seed(0)

logger = logging.getLogger(__name__)  # pylint: disable=locally-disabled, invalid-name

# Keras Parameters
MULTIPROCESSING = False
WORKERS = 1
MAX_QUEUE_SIZE = 100  # default=10

# Prepossessing parameters
SEQUENCE_LENGTH = 400

# Word2Vec parameters (see train_word2vec)
EMBEDDING_DIM = 300
MIN_WORD_COUNT = 1
CONTEXT = 10

# declare preprocessing params for saving them later
PREPROCESSING_PARAMS = {
    'sequence_length': SEQUENCE_LENGTH,
    'embedding_dim': EMBEDDING_DIM,
    'min_word_count': MIN_WORD_COUNT,
    'context': CONTEXT
}

params_run_A = {
    # Model Hyperparameters
    'filter_sizes': [(3, 4, 5)],
    'num_filters': [50, 100, 200],
    'dropout_prob': [(0, 0.5), (0, 0.6), (0, 0.7), (0, 0.8)],
    'hidden_dims': [[50]],

    # Training parameters
    'batch_size': [64],
    'num_epochs': [10],
    'optimizer': ['adam'],
    'lr': [0.01, 0.001, 0.0001],
    'decay': [0.0, 0.1, 0.2, 0.3]
}

params_run_B = {
    # Model Hyperparameters
    'filter_sizes': [(3, 4, 5), (3, 8)],
    'num_filters': [50, 100, 200],
    'dropout_prob': [(0, 0.5), (0, 0.6), (0, 0.7), (0, 0.8)],
    'hidden_dims': [[50], [50, 50, 50, 50]],

    # Training parameters
    'batch_size': [64],
    'num_epochs': [10],
    'optimizer': ['adam'],
    'lr': [0.001, 0.0001],
    'decay': [0.0]
}

params_run_STACKOVERFLOW = {
    # Model Hyperparameters
    'filter_sizes': [(3, 4, 5)],
    'num_filters': [50],
    'dropout_prob': [(0, 0.5), (0, 0.6), (0, 0.7), (0, 0.8)],
    'hidden_dims': [[50]],

    # Training parameters
    'batch_size': [64],
    'num_epochs': [10],
    'optimizer': ['adam'],
    'lr': [0.001, 0.0001],
    'decay': [0.0]
}

MODEL_PARAMS = params_run_STACKOVERFLOW


def _get_callbacks(run_dir):
    es = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, verbose=0, mode='auto')

    checkpoint_path = join(run_dir, 'model-weights.hdf5')
    checkpoints = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0,
                                  save_best_only=True, mode='auto', period=1)

    callbacks = namedtuple('callbacks', ['early_stopping', 'checkpoints'])
    return callbacks(es, checkpoints)


def preprocess(docs):
    padded = []
    for doc in docs:
        padded.append(preprocessing.pad_sequence(doc, max_length=SEQUENCE_LENGTH))
    return padded


def load_data(community_name):
    dataset = namedtuple('dataset', ['X_train', 'y_train', 'train_ids', 'X_val', 'y_val',
                                     'val_ids', 'X_test', 'y_test', 'test_ids', 'vocabulary_inv'])

    if community_name == 'CNN_DEBUG':
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000, start_char=None,
                                                              oov_char=None, index_from=None)

        x_train = sequence.pad_sequences(x_train, maxlen=SEQUENCE_LENGTH,
                                         padding="post", truncating="post")
        x_test = sequence.pad_sequences(x_test, maxlen=SEQUENCE_LENGTH,
                                        padding="post", truncating="post")

        vocabulary = imdb.get_word_index()
        vocabulary_inv = dict((v, k) for k, v in vocabulary.items())
        vocabulary_inv[0] = "<PAD/>"

        train_ids = np.array(range(0, 10))
        val_ids = np.array(range(10, 20))
        test_ids = np.array(range(20, 30))
        return dataset(x_train[:10], y_train[:10], train_ids,
                       x_test[:10], y_test[:10], val_ids,
                       x_test[10:20], y_test[10:20], test_ids,
                       vocabulary_inv)

    X_train, y_train, train_ids, X_val, y_val, val_ids, X_test, y_test, test_ids = preprocessing \
        .load_community(community_name, preprocess=True, min_df=3, with_dev=True)

    logger.info('Pad documents...')
    X_train = preprocess(X_train)
    X_val = preprocess(X_val)
    X_test = preprocess(X_test)

    logger.info('Build vocabulary...')
    vocabulary, vocabulary_inv = preprocessing.build_vocab(X_train + X_val + X_test)
    logger.info('Map vocabulary...')
    X_train = preprocessing.map_vocabulary(X_train, vocabulary)
    X_val = preprocessing.map_vocabulary(X_val, vocabulary)
    X_test = preprocessing.map_vocabulary(X_test, vocabulary)

    return dataset(X_train, y_train, train_ids,
                   X_val, y_val, val_ids,
                   X_test, y_test, test_ids,
                   vocabulary_inv)


def build_model(dropout_prob, filter_sizes, num_filters, hidden_dims, optimizer, lr, decay):
    input_shape = (SEQUENCE_LENGTH, EMBEDDING_DIM)
    model_input = Input(shape=input_shape)

    z = model_input
    z = Dropout(dropout_prob[0])(z)

    # Convolutional block
    conv_blocks = []
    for sz in filter_sizes:
        conv = Convolution1D(filters=num_filters,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=1)(z)
        conv = MaxPooling1D(pool_size=2)(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    z = Dropout(dropout_prob[1])(z)

    for dim in hidden_dims:
        z = Dense(dim, activation="relu")(z)
    model_output = Dense(1, activation="sigmoid")(z)

    model = Model(model_input, model_output)

    opt = None
    if optimizer == 'adam':
        opt = Adam(lr=lr, decay=decay)
    else:
        raise ValueError('Unknown optimizer {}'.format(optimizer))

    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model


def make_run(run_dir, config, dataset, embedding_weights, run_name):
    model = build_model(**cnn_util.filter_func_args(config, build_model))
    cnn_util.save_summary(run_dir, model)

    X_train, y_train = dataset.X_train, dataset.y_train
    X_val, y_val = dataset.X_val, dataset.y_val
    X_test, y_test, test_ids = dataset.X_test, dataset.y_test, dataset.test_ids

    # Train the model
    train_generator = CNNStaticGenerator(X_train, y_train, config['batch_size'], embedding_weights)
    val_generator = CNNStaticGenerator(X_val, y_val, config['batch_size'], embedding_weights)

    callbacks = _get_callbacks(run_dir)

    history = model.fit_generator(train_generator, epochs=config['num_epochs'],
                                  validation_data=val_generator, verbose=2,
                                  callbacks=list(callbacks),
                                  use_multiprocessing=MULTIPROCESSING,
                                  workers=WORKERS,
                                  max_queue_size=MAX_QUEUE_SIZE)

    cnn_util.save_history(run_dir, history.history)
    cnn_util.save_best_epoch(run_dir, callbacks.early_stopping)

    model.load_weights(join(run_dir, 'model-weights.hdf5'))
    evaluate_model(run_dir, model, X_val, y_val, config, embedding_weights, set_name='val')
    y_pred, y_pred_proba = evaluate_model(run_dir, model, X_test, y_test, config,
                                          embedding_weights, set_name='test')
    cnn_util.save_predictions(np.stack([test_ids, y_pred, y_pred_proba], axis=1), run_dir, run_name)


def main(args, metadata_dir, execution_id):
    cnn_util.save_config(metadata_dir, PREPROCESSING_PARAMS, 'preprocessing-params')

    # Data Preparation
    logger.info("Load data...")
    dataset = load_data(args.community)
    X_train, X_val, X_test = dataset.X_train, dataset.X_val, dataset.X_test
    vocabulary_inv = dataset.vocabulary_inv

    logger.info("X_train shape: %s", X_train.shape)
    logger.info("X_val shape: %s", X_val.shape)
    logger.info("X_test shape: %s", X_test.shape)
    logger.info("Vocabulary Size: %d", len(vocabulary_inv))

    embedding_weights = w2v.train_word2vec(np.vstack((X_train, X_val, X_test)), vocabulary_inv,
                                           args.community, num_features=EMBEDDING_DIM,
                                           min_word_count=MIN_WORD_COUNT, context=CONTEXT)

    grid = list(ParameterGrid(MODEL_PARAMS))
    run_names = ['run_{}'.format(i) for i in range(len(grid))]

    failed_runs = []
    logger.info('Start testing "%d" configurations', len(grid))
    for config, run_name in zip(grid, run_names):
        try:
            logger.info('%s...', run_name)
            run_dir = join(metadata_dir, run_name)
            os.makedirs(run_dir, exist_ok=True)
            cnn_util.save_config(run_dir, config)

            make_run(run_dir, config, dataset, embedding_weights, run_name)
        except Exception as e:
            logging.exception('Exception during run %s', run_name)
            failed_runs.append(run_name)

    if failed_runs:
        logger.warning('Failed runs: %s', str(failed_runs))
        run_names = [run for run in run_names if run not in failed_runs]

    if run_names:
        cnn_util.consolidate_runs(args, metadata_dir, run_names, execution_id)
    else:
        logger.error('All runs failed!')


def evaluate_model(run_dir, model, X, y_true, config, embedding_weights, set_name):
    gen = CNNStaticGenerator(X, y_true, config['batch_size'], embedding_weights, return_y=False)

    y_pred_proba = model.predict_generator(gen, use_multiprocessing=MULTIPROCESSING,
                                           workers=WORKERS, max_queue_size=MAX_QUEUE_SIZE)[:, 0]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    class_names = preprocessing.LABEL_ENCODER.classes_
    scores = evaluation._evaluate(y_true, y_pred, y_pred_proba, class_names)
    file_name = '{}_results.csv'.format(set_name)
    pd.DataFrame(scores, index=[0]).to_csv(join(run_dir, file_name), index=False)
    return y_pred, y_pred_proba


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--community", help="community name", type=str, default='CNN_DEBUG')
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()
    run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
    metadata_dir = join(dirname(__file__), '../../output/cnn/{}/{}'.format(args.community, run_id))
    os.makedirs(metadata_dir, exist_ok=True)

    log_filename = join(metadata_dir, 'training.log')
    logging.getLogger('gensim').setLevel(logging.WARNING)
    logging.config.fileConfig(join(dirname(__file__), '../logging_file.ini'),
                              disable_existing_loggers=False,
                              defaults={'logfilename': log_filename})
    main(args, metadata_dir, run_id)
