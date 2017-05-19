from typing import Set, Dict, List
import random
import numpy as np
import tensorflow as tf
import os
import pickle

from qanta.util.io import safe_open
from qanta.config import conf
from qanta import logging


log = logging.get(__name__)


def create_embeddings(vocab: Set[str], expand_glove=False, mask_zero=False):
    """
    Create embeddings
    :param vocab: words in the vocabulary
    :param expand_glove: Whether or not to expand embeddings past pre-trained ones
    :param mask_zero: if True, then 0 is reserved as a sequence length mask (distinct from UNK)
    :return: 
    """
    embeddings = []
    embedding_lookup = {}
    with open(conf['word_embeddings']) as f:
        i = 0
        line_number = 0
        n_bad_embeddings = 0
        if mask_zero:
            emb = np.zeros((conf['embedding_dimension']))
            embeddings.append(emb)
            embedding_lookup['MASK'] = i
            i += 1
        for l in f:
            splits = l.split()
            word = splits[0]
            if word in vocab:
                try:
                    emb = [float(n) for n in splits[1:]]
                except ValueError:
                    n_bad_embeddings += 1
                    continue
                embeddings.append(emb)
                embedding_lookup[word] = i
                i += 1
            line_number += 1
        n_embeddings = i
        log.info('Loaded {} embeddings'.format(n_embeddings))
        log.info('Encountered {} bad embeddings that were skipped'.format(n_bad_embeddings))
        mean_embedding = np.array(embeddings).mean(axis=0)
        if expand_glove:
            embed_dim = len(embeddings[0])
            words_not_in_glove = vocab - set(embedding_lookup.keys())
            for w in words_not_in_glove:
                emb = np.random.rand(embed_dim) * .08 * 2 - .08
                embeddings.append(emb)
                embedding_lookup[w] = i
                i += 1

            log.info('Initialized an additional {} embeddings not in dataset'.format(i - n_embeddings))

        log.info('Total number of embeddings: {}'.format(i))

        embeddings = np.array(embeddings)
        embed_with_unk = np.vstack([embeddings, mean_embedding])
        embedding_lookup['UNK'] = i
        return embed_with_unk, embedding_lookup


def make_layer(i: int, in_tensor, n_out, op, n_in=None,
               dropout_prob=None,
               batch_norm=False,
               batch_is_training=None,
               tf_histogram=False,
               reuse=None):
    with tf.variable_scope('layer' + str(i), reuse=reuse):
        if batch_norm and batch_is_training is None:
            raise ValueError('if using batch norm then passing a training placeholder is required')
        w = tf.get_variable('w', (in_tensor.get_shape()[1] if n_in is None else n_in, n_out),
                            dtype=tf.float32)
        if dropout_prob is not None:
            w = tf.nn.dropout(w, keep_prob=1 - dropout_prob)
        b = tf.get_variable('b', n_out, dtype=tf.float32)
        out = tf.matmul(in_tensor, w) + b
        if batch_norm:
            out = tf.contrib.layers.batch_norm(
                out, center=True, scale=True, is_training=batch_is_training, scope='bn', fused=True)
        out = (out if op is None else op(out))
        if tf_histogram:
            tf.summary.histogram('weights', w)
            tf.summary.histogram('biases', b)
            tf.summary.histogram('activations', out)
        return out, w


def parametric_relu(_x):
    alphas = tf.get_variable(
        'alpha',
        _x.get_shape()[-1],
        initializer=tf.constant_initializer(0.0),
        dtype=tf.float32
    )
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg


def convert_text_to_embeddings_indices(words: List[str], embedding_lookup: Dict[str, int]):
    """
    Convert a list of word tokens to embedding indices
    :param words: 
    :param embedding_lookup: 
    :param mentions: 
    :return: 
    """
    w_indices = []
    for w in words:
        if w in embedding_lookup:
            w_indices.append(embedding_lookup[w])
        else:
            w_indices.append(embedding_lookup['UNK'])
    return w_indices


def tf_format(x_data: List[List[int]], max_len: int, zero_index: int):
    """
    Pad with elements until it has max_len or shorten it until it has max_len. When padding insert
    the zero index so it doesn't contribute anything
    :param x_data:
    :param max_len:
    :param zero_index:
    :return:
    """
    for i in range(len(x_data)):
        row = x_data[i]
        while len(row) < max_len:
            row.append(zero_index)
        x_data[i] = x_data[i][:max_len]
    return x_data


def create_load_embeddings_function(we_tmp_target, we_target, logger):
    def load_embeddings(vocab=None, root_directory='', expand_glove=False, mask_zero=False):
        if os.path.exists(we_tmp_target):
            logger.info('Loading word embeddings from tmp cache')
            with safe_open(we_tmp_target, 'rb') as f:
                return pickle.load(f)
        elif os.path.exists(os.path.join(root_directory, we_target)):
            logger.info('Loading word embeddings from restored cache')
            with safe_open(os.path.join(root_directory, we_target), 'rb') as f:
                return pickle.load(f)
        else:
            if vocab is None:
                raise ValueError('To create fresh embeddings a vocab is needed')
            with safe_open(we_tmp_target, 'wb') as f:
                logger.info('Creating word embeddings and saving to cache')
                embed_and_lookup = create_embeddings(vocab, expand_glove=expand_glove, mask_zero=mask_zero)
                pickle.dump(embed_and_lookup, f)
                return embed_and_lookup
    return load_embeddings


def create_batches(batch_size,
                   x_data: np.ndarray, y_data: np.ndarray, x_lengths: np.ndarray,
                   pad=False, shuffle=True):
    if type(x_data) != np.ndarray or type(y_data) != np.ndarray:
        raise ValueError('x and y must be numpy arrays')
    if len(x_data) != len(y_data):
        raise ValueError('x and y must have the same dimension')
    n = len(x_data)
    order = list(range(n))
    if shuffle:
        np.random.shuffle(order)
    for i in range(0, n, batch_size):
        if len(order[i:i + batch_size]) == batch_size:
            x_batch = x_data[order[i:i + batch_size]]
            y_batch = y_data[order[i:i + batch_size]]
            x_batch_lengths = x_lengths[order[i:i + batch_size]]
            yield x_batch, y_batch, x_batch_lengths
        elif pad:
            size = len(order[i:i + batch_size])
            x_batch = np.vstack((
                x_data[order[i:i + batch_size]],
                np.zeros((batch_size - size, x_data.shape[1])))
            )
            y_batch = np.hstack((
                y_data[order[i:i + batch_size]],
                np.zeros((batch_size - size,)))
            )
            x_batch_lengths = np.hstack((
                x_lengths[order[i:i + batch_size]],
                np.zeros((batch_size - size,)))
            )
            yield x_batch, y_batch, x_batch_lengths
        else:
            break


def batch_generator(x: List[np.ndarray], y: np.ndarray, batch_size, n_batches, shuffle=True):
    """
    Create batches of size batch_size for each numpy array in data for n_batches then repeat
    """
    n_rows = len(y)
    for i, arr in enumerate(x):
        if type(arr) != np.ndarray:
            raise ValueError('An element of x in position {} is not a numpy array'.format(i))
    if type(y) != np.ndarray:
        raise ValueError('y must be a numpy array')
    if len(x) == 0:
        raise ValueError('There is no x data to batch')
    if n_rows == 0:
        raise ValueError('There are no rows to batch')
    if n_batches == 0:
        raise ValueError('Cannot iterate on batches when n_batches=0')
    if type(n_batches) != int:
        raise ValueError('n_batches must be of type int but was "{}"'.format(type(n_batches)))
    while True:
        order = np.array(list(range(n_rows)))
        if shuffle:
            np.random.shuffle(order)
        for i in range(n_batches):
            i_start = int(i * batch_size)
            i_end = int((i + 1) * batch_size)
            if len(x) > 1:
                x_batch = []
                for arr in x:
                    x_batch.append(arr[order[i_start:i_end]])
            else:
                x_batch = x[0][order[i_start:i_end]]
            yield x_batch, y[order[i_start:i_end]]


def compute_n_classes(labels: List[str]):
    return len(set(labels))


def compute_max_len(training_data):
    return max([len(' '.join(sentences).split()) for sentences in training_data[0]])


def compute_lengths(x_data):
    return np.array([max(1, len(x)) for x in x_data])
