import time
import pickle
import numpy as np

from qanta import logging
from qanta.guesser.util import gen_util
from qanta.util.io import safe_open
from qanta.guesser.classify.learn_classifiers import evaluate, compute_vectors
from qanta.guesser.util.adagrad import Adagrad
from qanta.guesser.util.functions import relu, drelu
from qanta.util.constants import (DEEP_WE_TARGET, DEEP_DAN_PARAMS_TARGET, DEEP_TRAIN_TARGET,
                                  DEEP_DEV_TARGET, DEEP_DAN_TRAIN_OUTPUT, DEEP_DAN_DEV_OUTPUT)

log = logging.get(__name__)


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
