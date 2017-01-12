from numpy import *
import pickle

from qanta import logging
from qanta.guesser.util.dataset import Dataset, get_or_make_id_map
from qanta.util.io import safe_open


log = logging.get(__name__)


def create():
    vec_file = open('data/external/deep/glove.840B.300d.txt')
    all_vocab = {}
    log.info('loading vocab...')
    # TODO: Fix this hack
    wmap = get_or_make_id_map([Dataset.QUIZ_BOWL, Dataset.WIKI])

    for line in vec_file:
        split = line.split()
        word = " ".join(split[:-300])
        if word not in wmap:
            continue
        x = wmap[word]
        all_vocab[word] = array(split[-300:])
        all_vocab[word] = all_vocab[word].astype(float)

    log.info("wmap: {0} all_vocab: {1}".format(len(wmap), len(all_vocab)))
    d = len(all_vocab['the'])

    We = empty((d, len(wmap)))

    log.info('creating We for {0} words'.format(len(wmap)))
    unknown = []

    offset = len(wmap)
    log.info('offset = {0}'.format(offset))

    for word in wmap:
        try:
            We[:, wmap[word]] = all_vocab[word]
        except KeyError:
            unknown.append(word)
            log.info('unknown: {0}'.format(word))
            # initialize unknown words with unknown token
            We[:, wmap[word]] = all_vocab['unknown']

    log.info('unknown: {0}'.format(len(unknown)))
    log.info('We shape: {0}'.format(We.shape))

    log.info('dumping...')
    with safe_open('output/deep/We', 'wb') as f:
        pickle.dump(We, f, protocol=pickle.HIGHEST_PROTOCOL)
