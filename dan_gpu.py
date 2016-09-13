import theano, lasagne, pickle, time, argparse                                                 
import numpy as np
import theano.tensor as T     
from collections import OrderedDict, Counter
from unidecode import unidecode
from string import ascii_lowercase, punctuation
from nltk.corpus import stopwords




def iterate_minibatches(inputs, masks, labels, batch_size, shuffle=False):
    assert len(inputs) == len(labels)
    assert len(inputs) == len(masks)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], masks[excerpt], labels[excerpt]

def validate(name, val_fn, fold, batch_s):
    corr = 0
    corr_200 = 0
    total = 0
    c1 = Counter()
    sents, masks, labels = fold
    for s, m, l in iterate_minibatches(sents, masks, labels, batch_s, shuffle=False):
        preds = val_fn(s, m)
        preds = np.argsort(-preds, axis=1)[:, :50]

        for i in range(preds.shape[0]):
            pred = preds[i]
            if pred[0] == l[i]:
                corr += 1
            if l[i] in set(pred):
                corr_200 += 1
            total += 1

            c1[pred[0]] += 1

    lstring = 'fold:%s, corr:%d, corr50:%d, total:%d, acc:%f, recall50:%f' %\
        (name, corr, corr_200, total, float(corr) / float(total), float(corr_200) / float(total))
    print(lstring)
    print([(rev_ans_dict[w],count) for (w,count) in c1.most_common(10)])
    return lstring

def validate_no_batch(name, val_fn, fold, batch_s):
    corr = 0
    corr_200 = 0
    total = 0
    c1 = Counter()
    sents, masks, labels = fold
#    for s, m, l in iterate_minibatches(sents, masks, labels, batch_s, shuffle=False):
    preds = val_fn(sents, masks)
    preds = np.argsort(-preds, axis=1)[:, :50]

    for i in range(preds.shape[0]):
        pred = preds[i]
        if pred[0] == labels[i]:
            corr += 1
        if labels[i] in set(pred):
            corr_200 += 1
        total += 1

        c1[pred[0]] += 1

    lstring = 'fold:%s, corr:%d, corr200:%d, total:%d, acc:%f, recall50:%f' %\
        (name, corr, corr_200, total, float(corr) / float(total), float(corr_200) / float(total))
    print(lstring)
    print([(rev_ans_dict[w],count) for (w,count) in c1.most_common(10)])
    return lstring

class SumLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, **kwargs):
        super(SumLayer, self).__init__(incomings, **kwargs)

    def get_output_for(self, inputs, **kwargs):
        return T.sum(inputs[0] * inputs[1][:, :, None], axis=1, dtype=theano.config.floatX) / inputs[1].shape[1]

    # batch_size x d
    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], input_shapes[0][2])

def build_lstm(sents, masks, labels, len_voc, d_word, d_van,
    num_labels, max_len, We, freeze=True, eps=1e-6, lr=0.1, rho=1e-5):
    

    # define network
    l_in = lasagne.layers.InputLayer(shape=(None, max_len), input_var=sents, )
    l_mask = lasagne.layers.InputLayer(shape=(None, max_len), input_var=masks)
    l_emb = lasagne.layers.EmbeddingLayer(l_in, len_voc, d_word, W=We) #Errored with W=We

    # now feed sequences of spans into VAN
    # l_lstm = lasagne.layers.LSTMLayer(l_emb, d_van, mask_input=l_mask, )
    l_lstm = SumLayer([l_emb, l_mask])
    # freeze embeddings
    if freeze:
        l_emb.params[l_emb.W].remove('trainable')

    # now predict
    # l_forward_slice = lasagne.layers.SliceLayer(l_lstm, -1, 1)
    
    l_hid1 = lasagne.layers.DenseLayer(l_lstm, num_units=d_word, nonlinearity=lasagne.nonlinearities.rectify)

    l_hid2 = lasagne.layers.DenseLayer(l_hid1, num_units=d_word, nonlinearity=lasagne.nonlinearities.rectify)
    #l_hid3 = lasagne.layers.DenseLayer(l_hid2, num_units=d_word, nonlinearity=lasagne.nonlinearities.rectify)
    l_out = lasagne.layers.DenseLayer(l_hid2, num_units=num_labels,\
        nonlinearity=lasagne.nonlinearities.softmax)

    # objective computation
    preds = lasagne.layers.get_output(l_out)
    loss = T.sum(lasagne.objectives.categorical_crossentropy(preds, labels))
    loss += rho * sum(T.sum(l ** 2) for l in lasagne.layers.get_all_params(l_out))
    all_params = lasagne.layers.get_all_params(l_out)

    updates = lasagne.updates.adam(loss, all_params, learning_rate=lr)

    train_fn = theano.function([sents, masks, labels], [preds, loss], updates=updates)
    val_fn = theano.function([sents, masks], preds)
    debug_fn = theano.function([sents, masks], lasagne.layers.get_output(l_lstm))
    return train_fn, val_fn, debug_fn, l_out

def load_model(path= 'output/deep/dan_gpu_param_values.pkl'):
    train_fn, val_fn, debug_fn, l_out = build_lstm(sents, masks, labels, len_voc, d_word, d_hid,num_labels, max_len, We.T, freeze=freeze, lr=lr)
    params = pickle.load(open(path, 'rb'))
    lasagne.layers.set_all_param_values(l_out, params)
    return train_fn, val_fn, debug_fn, l_out

def save_model(final_layer, path = 'output/deep/dan_gpu_param_values.pkl'):
    params = lasagne.layers.get_all_param_values(final_layer)
    pickle.dump(params, open(path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

def normalize(text):
    ENGLISH_STOP_WORDS = set(stopwords.words('english'))
    QB_STOP_WORDS = {"10", "ten", "points", "tenpoints", "one", "name", ",", ")", "``", "(", '"', ']',
                 '[', ":", "due", "!", "'s", "''", 'ftp'}
    STOP_WORDS = ENGLISH_STOP_WORDS | QB_STOP_WORDS
    valid_strings = set(ascii_lowercase) | set(str(x) for x in range(10)) | {' '}
    text = unidecode(text).lower().translate(str.maketrans(punctuation, ' ' * len(punctuation)))
    #text = PAREN_EXPRESSION.sub("", text)
    text = " ".join(x for x in text.split() if x not in STOP_WORDS)
    return ''.join(x for x in text if x in valid_strings)


def compute_features(text: str):

        # generate word vector lookups given normalized input text
        text = normalize(text)
        for ner in self.ners:
            text = text.replace(ner, ner.replace(' ', '_'))

        inds = []
        for w in text.split():
            if w in self.vdict:
                inds.append(self.vdict[w])

        if len(inds) > 0:
            # compute vector representation for question text
            np.zeros((1, len(inds))).astype('float32')
		    #av = np.average(L[:, inds], axis=1).reshape((self.d, 1))
            p3 = inds

        else:
            p3 = np.zeros((1, self.d))

        return p3.ravel().reshape(1, -1)

def compute_probs(self, text):
    curr_feats = self.compute_features(text)
    return self.predict_proba(curr_feats, mask)

# train qanta and save model
if __name__ == '__main__':

    # command line arguments
    parser = argparse.ArgumentParser(description='QANTA: a question answering neural network \
                                     with trans-sentential aggregation')
    parser.add_argument('-We', help='location of word embeddings', default='output/deep/We_py2')
    parser.add_argument('-d', help='word embedding dimension', type=int, default=300)
    parser.add_argument('-b', '--batch_size', help='adagrad minibatch size (ideal: 25 minibatches \
                        per epoch). for provided datasets, x for history and y for lit', type=int,\
                        default=100)
    parser.add_argument('-ep', '--num_epochs', help='number of training epochs, can also determine \
                         dynamically via validate method', type=int, default=61)
    parser.add_argument('-agr', '--adagrad_reset', help='reset sum of squared gradients after this many\
                         epochs', type=int, default=10)
    parser.add_argument('-lr', help='adagrad learning rate', type=float, default=0.01)
    parser.add_argument('-v', '--do_val', help='check performance on dev set after this many\
                         epochs', type=int, default=50)
    parser.add_argument('-o', '--output', help='desired location of output model', \
                         default='output/deep/params_dan_gpu')

    args = vars(parser.parse_args())
    d = args['d']

    # load data
    train = pickle.load(open('output/deep/train_py2', 'rb'))
    dev = pickle.load(open('output/deep/dev_py2', 'rb'))
    vocab, vdict = pickle.load(open('output/deep/vocab_py2', 'rb'))

    # make DAN GPU data
    folds = [train, dev]
    trgpu = []
    devgpu = []
    max_len = -1
    ans_dict = {}
    for index, fold in enumerate(folds):
      for qs, ans in fold:
            ans = ans[0]
            ans = vocab[ans]
            if ans not in ans_dict:
                ans_dict[ans] = len(ans_dict)
            history = []

            for dist in qs:
                sent = qs[dist]
                history += sent
                if index == 0:
                    trgpu.append((history, ans_dict[ans]))
                else:
                    devgpu.append((history, ans_dict[ans]))
                if max_len < len(history):
                    max_len = len(history)

    train = np.zeros((len(trgpu), max_len)).astype('int32')
    dev = np.zeros((len(devgpu), max_len)).astype('int32')
    trmask = np.zeros((len(trgpu), max_len)).astype('float32')
    devmask = np.zeros((len(devgpu), max_len)).astype('float32')
    trlabels = np.zeros(len(trgpu)).astype('int32')
    devlabels = np.zeros(len(devgpu)).astype('int32')
    We = pickle.load(open(args['We'], 'rb')).astype('float64')


    for i, (q, ans) in enumerate(trgpu):
        stop = len(q)
        train[i, :stop] = q
        trmask[i, :stop] = 1.
        trlabels[i] = ans
    for i, (q, ans) in enumerate(devgpu):
        stop = len(q)
        dev[i, :stop] = q
        devmask[i, :stop] = 1.
        devlabels[i] = ans

    len_voc = len(vocab)
    num_labels = len(ans_dict)
    d_word = 300
    d_hid = 300
    lr = np.float32(0.002)
    freeze = True
    batch_size = 256
    n_epochs = 50
    drop_prob = 0.75 #It's the inverse of dropping. Higher keeps more words
    print("Training size: %s, Dev size: %s, Maximum length: %s, Classes: %s, Vocab size: %s" %(len(trgpu), len(devgpu), max_len, num_labels, len_voc))

    # dump ans_dict for later
    rev_ans_dict = dict((v,k) for (k,v) in ans_dict.items())
    pickle.dump((ans_dict, rev_ans_dict), open('output/deep/ans_dict.pkl', 'wb'))
    log_file = 'output/deep/dan_log.txt'
    log = open(log_file, 'w')

    print('compiling graph...')

    # input theano vars
    sents = T.imatrix(name='sentence')
    masks = T.matrix(name='mask')
    labels = T.ivector('target')
    #train_fn, val_fn, debug_fn, final_layer = build_lstm(sents, masks, labels, len_voc, d_word, d_hid, 
    #    num_labels, max_len, We.T, freeze=freeze, lr=lr)
    train_fn, val_fn, debug_fn, final_layer = load_model()
    print('done compiling')
    
    # train network
    for epoch in range(n_epochs):
        cost = 0.
        start = time.time()
        num_batches = 0.
        for s, m, l in iterate_minibatches(train, trmask, trlabels, batch_size, shuffle=True):
            mask = np.random.binomial(1, drop_prob, (batch_size, max_len)).astype('float32')
            #input("Enter to continue")
            preds, loss = train_fn(s, m*mask, l)
            cost += loss
            num_batches += 1
            if num_batches % 500 == 0: print(num_batches)

        lstring = 'epoch:%d, cost:%f, time:%d' % \
            (epoch, cost / num_batches, time.time()-start )
        print(lstring)
        log.write(lstring + '\n')

        trperf = validate('train', val_fn, [train, trmask, trlabels], batch_size)
        devperf = validate('dev', val_fn, [dev, devmask, devlabels], batch_size)
        log.write(trperf + '\n')
        #log.write(devperf + '\n')
        log.flush()
        print('\n')

        # save params
        save_model(final_layer)

        #params = lasagne.layers.get_all_params(final_layer)
        #p_values = [p.get_value() for p in params]
        #p_dict = dict(zip([str(p) for p in params], p_values))
        #pickle.dump(params, open('output/deep/dan_gpu_params.pkl', 'wb'),
        #    protocol=pickle.HIGHEST_PROTOCOL)
