from numpy import *
import pickle, gzip, gc, glob, sys, os
from collections import Counter

# given a vocab (list of words), return a word embedding matrix
if __name__ == '__main__':

    vec_file = gzip.open('data/deep/glove.840B.300d.txt.gz', 'rb')
    all_vocab = {}
    print('loading vocab...')
    vocab, wmap = pickle.load(open('data/deep/vocab', 'rb'))

    for line in vec_file:
        split = line.split()
        try:
            word = split[0].decode("utf-8")
        except UnicodeDecodeError:
            print('Unicode error, skipping word')
        if word not in wmap:
            continue
        x = wmap[word]
        all_vocab[word] = array(split[1:])
        all_vocab[word] = all_vocab[word].astype(float)

    print(len(wmap), len(all_vocab))
    d = len(all_vocab['the'])

    We = empty((d, len(wmap)))

    print('creating We for ', len(wmap), ' words')
    unknown = []

    offset = len(wmap)
    print('offset = ', offset)

    for word in wmap:
        try:
            We[:, wmap[word]] = all_vocab[word]
        except KeyError:
            unknown.append(word)
            print('unknown: ', word)
            # initialize unknown words with unknown token
            We[:, wmap[word]] = all_vocab['unknown']

    print('unknown: ', len(unknown))
    print('We shape: ', We.shape)

    print('dumping...')
    pickle.dump( We, open('data/deep/We', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
