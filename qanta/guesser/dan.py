import time
import pickle
import numpy as np

from qanta.guesser.util import gen_util
from qanta.guesser.classify.learn_classifiers import evaluate
from qanta.guesser.util.adagrad import Adagrad
from qanta.guesser.util.functions import relu, drelu
from qanta.util.constants import (DEEP_WE_TARGET, DEEP_DAN_PARAMS_TARGET, DEEP_TRAIN_TARGET,
                                  DEEP_DEV_TARGET)


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


def train(batch_size=150, we_dimension=300, n_epochs=61, learning_rate=0.01, adagrad_reset=10):
    # load data
    train = pickle.load(open(DEEP_TRAIN_TARGET, 'rb'))
    dev = pickle.load(open(DEEP_DEV_TARGET, 'rb'))

    train_qs = train
    val_qs = dev
    print('total questions: ', len(train_qs))
    total = 0
    for qs, ans in train_qs:
        total += len(qs)
    print('total sentences: ', total)

    orig_We = pickle.load(open(DEEP_WE_TARGET, 'rb'))

    len_voc = orig_We.shape[1]
    print('vocab length: ', len_voc, ' We shape: ', orig_We.shape)

    # output log and parameter file destinations
    log_file = DEEP_DAN_PARAMS_TARGET.split('_')[0] + '_log'

    # generate params / We
    params = gen_util.init_params(we_dimension, deep=3)

    # add We matrix to params
    params += (orig_We, )
    r = gen_util.roll_params(params)

    dim = r.shape[0]
    print('parameter vector dimensionality:', dim)

    log = open(log_file, 'w')

    # minibatch adagrad training
    ag = Adagrad(r.shape, learning_rate)
    min_error = float('inf')

    print('step 1 of 2: training DAN (takes 2-3 hours)')
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
            lstring = 'epoch: ' + str(epoch) + ' batch_ind: ' + str(batch_ind) + ' error, '\
                      + str(err) + ' time = ' + str(time.time()-now) + ' sec'
            print(lstring)
            log.write(lstring + '\n')
            log.flush()

            epoch_error += err

        # done with epoch
        print(time.time() - ep_t)
        print('done with epoch ', epoch, ' epoch error = ', epoch_error, ' min error = ', min_error)
        lstring = 'done with epoch ' + str(epoch) + ' epoch error = ' + str(epoch_error) +\
                  ' min error = ' + str(min_error) + '\n\n'
        log.write(lstring)
        log.flush()

        # save parameters if the current model is better than previous best model
        if epoch_error < min_error:
            min_error = epoch_error
            print('saving model...')
            params = gen_util.unroll_params(r, we_dimension, len_voc, deep=3)
            pickle.dump(params, open(DEEP_DAN_PARAMS_TARGET, 'wb'))

        # reset adagrad weights
        if epoch % adagrad_reset == 0 and epoch != 0:
            ag.reset_weights()
    log.close()

    print('step 2 of 2: training classifier over all answers (takes 10-15 hours)')
    params = pickle.load(open(DEEP_DAN_PARAMS_TARGET, 'rb'))
    evaluate(train_qs, val_qs, params, we_dimension)
