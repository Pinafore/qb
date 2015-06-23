from numpy import *
import cPickle, gzip, gc, glob, sys, os
from collections import Counter

# given a vocab (list of words), return a word embedding matrix
if __name__ == '__main__':

    vec_file = gzip.open('data/deep/glove.840B.300d.txt.gz', 'r')
    all_vocab = {}
    print 'loading vocab...'
    vocab, wmap = cPickle.load(open('data/deep/vocab', 'rb'))

    for line in vec_file:
        split = line.split()
        try:
            x = wmap[split[0]]
            all_vocab[split[0]] = array(split[1:])
            all_vocab[split[0]] = all_vocab[split[0]].astype(float)
        except:
            pass

    print len(wmap), len(all_vocab)
    d = len(all_vocab['the'])

    We = empty( (d, len(wmap)) )

    print 'creating We for ', len(wmap), ' words'
    unknown = []

    offset = len(wmap)
    print 'offset = ', offset

    for word in wmap:
        try:
            We[:, wmap[word]] = all_vocab[word]
        except KeyError:
            unknown.append(word)
            print 'unknown: ', word
            # initialize unknown words with unknown token
            We[:, wmap[word]] = all_vocab['unknown']

    print 'unknown: ', len(unknown)
    print 'We shape: ', We.shape

    print 'dumping...'
    cPickle.dump( We, open('data/deep/We', 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
