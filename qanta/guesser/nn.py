from typing import Set, Dict, List
import random
import numpy as np
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


def convert_text_to_embeddings_indices(words: List[str], embedding_lookup: Dict[str, int], random_unk_prob=0):
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
            if random_unk_prob > 0 and random.random() < random_unk_prob:
                w_indices.append(embedding_lookup['UNK'])
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
    def load_embeddings(vocab=None, root_directory='', expand_glove=True, mask_zero=False):
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


def compute_n_classes(labels: List[str]):
    return len(set(labels))


def compute_max_len(training_data):
    return max([len(' '.join(sentences).split()) for sentences in training_data[0]])


def compute_lengths(x_data):
    return np.array([max(1, len(x)) for x in x_data])
