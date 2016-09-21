import time
import pickle
import numpy as np
import lasagne, theano
from theano import tensor as T
from collections import OrderedDict, Counter

from qanta import logging
from qanta.guesser.util import gen_util
from qanta.util.io import safe_open
from qanta.guesser.classify.learn_classifiers import evaluate, compute_vectors
from qanta.guesser.util.adagrad import Adagrad
from qanta.guesser.util.functions import relu, drelu
from qanta.util.constants import (DEEP_VOCAB_TARGET, DEEP_WE_TARGET, DEEP_DAN_PARAMS_TARGET, DEEP_TRAIN_TARGET,
                                  DEEP_DEV_TARGET, DEEP_DAN_TRAIN_OUTPUT, DEEP_DAN_DEV_OUTPUT)

log = logging.get(__name__)

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
    #print([(rev_ans_dict[w],count) for (w,count) in c1.most_common(10)])
    return lstring



class SumLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, **kwargs):
        super(SumLayer, self).__init__(incomings, **kwargs)

    def get_output_for(self, inputs, **kwargs):
        return T.sum(inputs[0] * inputs[1][:, :, None], axis=1, dtype=theano.config.floatX) / inputs[1].shape[1]

    # batch_size x d
    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0][0], input_shapes[0][2])


def objective_and_grad(data, params, d, len_voc, word_drop=0.3, rho=1e-5):
    params = gen_util.unroll_params(params, d, len_voc, deep=3)
    (W, b, W2, b2, W3, b3, L) = params
    grads = gen_util.init_grads(d, len_voc, deep=3)
    error_sum = 0.0

    for qs, ans in data:

        # answer vector
        comp = L[:, ans[0]].reshape((d, 1))
        history = []
        for dist in qs:

            sent = qs[dist]

            # compute average of non-dropped words
            history += sent
            curr_hist = []
            mask = np.random.rand(len(history)) > word_drop
            for index, keep in enumerate(mask):
                if keep:
                    curr_hist.append(history[index])

            # all examples must have at least one word
            if len(curr_hist) == 0:
                curr_hist = history
            if len(curr_hist) == 0:
                continue

            av = np.average(L[:, curr_hist], axis=1).reshape((d, 1))

            # apply non-linearity
            p = relu(W.dot(av) + b)
            p2 = relu(W2.dot(p) + b2)
            p3 = relu(W3.dot(p2) + b3)

            # compute error
            delta = np.zeros((d, 1))

            # randomly sample 100 wrong answers
            inds = np.array([w_ind for w_ind in np.random.randint(0, L.shape[1], 100)])
            wrong_ans = L[:, inds]
            prod = wrong_ans.T @ p3

            base = 1 - comp.T @ p3
            delta_base = -1 * comp.ravel()
            a = base + prod
            pos_inds = np.where(a > 0)[0]

            if len(pos_inds) > 0:
                error_sum += np.sum(a[pos_inds])
                dc = delta_base[:, np.newaxis] + wrong_ans[:, pos_inds]
                delta += np.sum(dc, axis=1).reshape((d, 1))

                # update correct / incorrect words w/ small learning rate
                grads[6][:, ans[0]] -= 0.0001 * p3.ravel()

            # backprop third layer
            delta_3 = drelu(p3) * delta
            grads[4] += delta_3 @ p2.T
            grads[5] += delta_3

            # backprop second layer
            delta_2 = drelu(p2) * W3.T.dot(delta_3)
            grads[2] += delta_2 @ p.T
            grads[3] += delta_2

            # backprop first layer
            delta_1 = drelu(p) * W2.T.dot(delta_2)
            grads[0] += delta_1 @ av.T
            grads[1] += delta_1
            grads[6][:, curr_hist] += W.T.dot(delta_1) / len(curr_hist)

    # L2 regularize
    for index in range(0, len(params)):
        error_sum += 0.5 * rho * np.sum(params[index] ** 2)
        grads[index] += rho * params[index]

    cost = error_sum / len(data)
    grad = gen_util.roll_params(grads) / len(data)

    return cost, grad

def build_dan(sents, masks, labels, len_voc, num_labels, We, d_word=300,
     max_len=100, freeze=True, eps=1e-6, lr=0.1, rho=1e-5):
    

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

def save_model(final_layer, path = DEEP_DAN_PARAMS_TARGET):
    params = lasagne.layers.get_all_param_values(final_layer)
    pickle.dump(params, open(path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def train_dan_gpu(batch_size=150, we_dimension=300, n_epochs=61, learning_rate=0.01):
    #args = vars(parser.parse_args())
    #d = args['d']

    # load data
    train = pickle.load(open(DEEP_TRAIN_TARGET, 'rb'))
    dev = pickle.load(open(DEEP_DEV_TARGET, 'rb'))
    vocab, vdict = pickle.load(open(DEEP_VOCAB_TARGET, 'rb'))

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
    We = pickle.load(open(DEEP_WE_TARGET, 'rb')).astype('float64')


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
    lr = np.float32(learning_rate)
    freeze = True
    #batch_size = 256
    #n_epochs = 50
    drop_prob = 0.75 #It's the inverse of dropping. Higher keeps more words
    print("Training size: %s, Dev size: %s, Maximum length: %s, Classes: %s, Vocab size: %s" %(len(trgpu), len(devgpu), max_len, num_labels, len_voc))

    # dump ans_dict for later
    rev_ans_dict = dict((v,k) for (k,v) in ans_dict.items())
    pickle.dump((ans_dict, rev_ans_dict), open(C.CLASS_LABEL_TARGET, 'wb'))

    log_file = 'output/deep/dan_log.txt'
    log = open(log_file, 'w')

    print('compiling graph...')

    # input theano vars
    sents = T.imatrix(name='sentence')
    masks = T.matrix(name='mask')
    labels = T.ivector('target')
    train_fn, val_fn, debug_fn, final_layer = build_dan(sents, masks, labels, len_voc, num_labels, We=We.T)

    # old method call build_dan(sents, masks, labels, len_voc, d_word, d_hid,          num_labels, max_len, We.T, freeze=freeze, lr=lr)
    #train_fn, val_fn, debug_fn, final_layer = load_model()
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
    


def train_dan(batch_size=150, we_dimension=300, n_epochs=61, learning_rate=0.01, adagrad_reset=10):
    with open(DEEP_TRAIN_TARGET, 'rb') as f:
        train_qs = pickle.load(f)

    log.info('total questions: {0}'.format(len(train_qs)))
    total = 0
    for qs, ans in train_qs:
        total += len(qs)
    log.info('total sentences: {0}'.format(total))

    with open(DEEP_WE_TARGET, 'rb') as f:
        orig_We = pickle.load(f)

    len_voc = orig_We.shape[1]
    log.info('vocab length: {0} We shape: {1}'.format(len_voc, orig_We.shape))

    # generate params / We
    params = gen_util.init_params(we_dimension, deep=3)

    # add We matrix to params
    params += (orig_We, )
    r = gen_util.roll_params(params)

    dim = r.shape[0]
    log.info('parameter vector dimensionality: {0}'.format(dim))

    # minibatch adagrad training
    ag = Adagrad(r.shape, learning_rate)
    min_error = float('inf')

    log.info('step 1 of 2: training DAN (takes 2-3 hours)')
    for epoch in range(0, n_epochs):
        # create mini-batches
        np.random.shuffle(train_qs)
        batches = [train_qs[x: x + batch_size] for x in list(range(0, len(train_qs), batch_size))]

        epoch_error = 0.0
        ep_t = time.time()

        for batch_ind, batch in enumerate(batches):
            now = time.time()
            err, grad = objective_and_grad(batch, r, we_dimension, len_voc)
            update = ag.rescale_update(grad)
            r -= update
            lstring = 'epoch: {0} batch_ind: {1} error, {2} time = {3}'.format(
                epoch, batch_ind, err, time.time() - now)
            log.info(lstring)
            epoch_error += err

        # done with epoch
        log.info(str(time.time() - ep_t))
        log.info('done with epoch {0} epoch error = {1} min error = {2}'.format(
            epoch, epoch_error, min_error))

        # save parameters if the current model is better than previous best model
        if epoch_error < min_error:
            min_error = epoch_error
            log.info('saving model...')
            params = gen_util.unroll_params(r, we_dimension, len_voc, deep=3)
            with safe_open(DEEP_DAN_PARAMS_TARGET, 'wb') as f:
                pickle.dump(params, f)

        # reset adagrad weights
        if epoch % adagrad_reset == 0 and epoch != 0:
            ag.reset_weights()


def compute_classifier_input(we_dimensions=300):
    # Load training data
    with open(DEEP_TRAIN_TARGET, 'rb') as f:
        train_qs = pickle.load(f)
    # Load dev data
    with open(DEEP_DEV_TARGET, 'rb') as f:
        val_qs = pickle.load(f)
    # Load trained_DAN parameters
    with open(DEEP_DAN_PARAMS_TARGET, 'rb') as f:
        params = pickle.load(f)
    
    # Compute training, dev classifier vectors using DAN
    train_vector, test_vector = compute_vectors(train_qs, val_qs, params, we_dimensions)
    
    # Format training vector
    train_feats = []
    train_labels = []
    for e in train_vector:
        train_feats.append(e[0])
        train_labels.append(e[1])
    train_formatted = (train_feats, train_labels)
    
    # Format dev vector
    test_feats = []
    test_labels = []
    for e in test_vector:
        test_feats.append(e[0])
        test_labels.append(e[1])
    test_formatted = (test_feats, test_labels)
    
    # Save
    with safe_open(DEEP_DAN_TRAIN_OUTPUT, 'wb') as f:
        pickle.dump(train_formatted, f, protocol=pickle.HIGHEST_PROTOCOL)
    with safe_open(DEEP_DAN_DEV_OUTPUT, 'wb') as f:
        pickle.dump(test_formatted, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info('Classifier train/dev vectors computed using DAN')


def train_classifier():
    log.info('step 2 of 2: training classifier over all answers')
    with open(DEEP_DAN_TRAIN_OUTPUT, 'rb') as f:
        train_formatted = pickle.load(f)
    with open(DEEP_DAN_DEV_OUTPUT, 'rb') as f:
        dev_formatted = pickle.load(f)
    evaluate(train_formatted, dev_formatted)
    log.info('finished training and saving classifier')
