from numpy import *
import pickle


def main():
    vec_file = open('data/external/deep/glove.840B.300d.txt')
    all_vocab = {}
    print('loading vocab...')
    vocab, wmap = pickle.load(open('output/deep/vocab', 'rb'))

    for line in vec_file:
        split = line.split()
        word = split[0]
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
    pickle.dump(We, open('output/deep/We', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

# given a vocab (list of words), return a word embedding matrix
if __name__ == '__main__':
    main()
