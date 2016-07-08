from numpy import *
import pickle

from qanta import logging


log = logging.get(__name__)


def create():
    vec_file = open('data/external/deep/glove.840B.300d.txt')
    all_vocab = {}
    log.info('loading vocab...')
    vocab, wmap = pickle.load(open('output/deep/vocab', 'rb'))

    for line in vec_file:
        split = line.split()
        word = split[0]
        if word not in wmap:
            continue
        x = wmap[word]
        all_vocab[word] = array(split[1:])
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
    pickle.dump(We, open('output/deep/We', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
