from numpy import *
from future.builtins import range
from guesser.util.gen_util import *
from guesser.util.math_util import *
from guesser.classify.learn_classifiers import evaluate
from guesser.util.adagrad import Adagrad
import time, argparse
from util.imports import pickle

# does both forward and backprop
def objective_and_grad(data, params, d, len_voc, word_drop=0.3, rho=1e-5):

    params = unroll_params(params, d, len_voc, deep=3)
    (W, b, W2, b2, W3, b3, L) = params
    grads = init_grads(d, len_voc, deep=3)
    error_sum = 0.0

    for qs, ans in data:

        # answer vector
        comp = L[:, ans[0]].reshape((d, 1))
        prev_sum = zeros((d, 1))
        history = []

        for dist in qs:

            sent = qs[dist]

            # compute average of non-dropped words
            history += sent
            curr_hist = []
            mask = random.rand(len(history)) > word_drop
            for index, keep in enumerate(mask):
                if keep:
                    curr_hist.append(history[index])

            # all examples must have at least one word
            if len(curr_hist) == 0:
                curr_hist = history
            if len(curr_hist) == 0:
                continue

            av = average(L[:, curr_hist], axis=1).reshape((d, 1))

            # apply non-linearity
            p = relu(W.dot(av) + b)
            p2 = relu(W2.dot(p) + b2)
            p3 = relu(W3.dot(p2) + b3)

            # compute error
            delta = zeros((d, 1))

            # randomly sample 100 wrong answers
            inds = array([w_ind for w_ind in random.randint(0, L.shape[1], 100)])
            wrong_ans = L[:, inds]
            prod = wrong_ans.T.dot(p3)

            base = 1 - comp.T.dot(p3)
            delta_base = -1 * comp.ravel()
            a = base + prod
            pos_inds = where(a>0)[0]

            if len(pos_inds) > 0:
                error_sum += sum(a[pos_inds])
                dc = delta_base[:, newaxis] + wrong_ans[:, pos_inds]
                delta += sum(dc, axis=1).reshape((d, 1))

                # update correct / incorrect words w/ small learning rate
                grads[6][:, ans[0]] -= 0.0001 * p3.ravel()

            # backprop third layer
            delta_3 = drelu(p3) * delta
            grads[4] += delta_3.dot(p2.T)
            grads[5] += delta_3

            # backprop second layer
            delta_2 = drelu(p2) * W3.T.dot(delta_3)
            grads[2] += delta_2.dot(p.T)
            grads[3] += delta_2

            # backprop first layer
            delta_1 = drelu(p) * W2.T.dot(delta_2)
            grads[0] += delta_1.dot(av.T)
            grads[1] += delta_1
            grads[6][:, curr_hist] += W.T.dot(delta_1) / len(curr_hist)

    # L2 regularize
    for index in range(0, len(params)):
        error_sum += 0.5 * rho * sum(params[index] ** 2)
        grads[index] += rho * params[index]

    cost = error_sum / len(data)
    grad = roll_params(grads) / len(data)

    return cost, grad

# train qanta and save model
if __name__ == '__main__':

    # command line arguments
    parser = argparse.ArgumentParser(description='QANTA: a question answering neural network \
                                     with trans-sentential aggregation')
    parser.add_argument('-We', help='location of word embeddings', default='data/deep/We')
    parser.add_argument('-d', help='word embedding dimension', type=int, default=300)
    parser.add_argument('-b', '--batch_size', help='adagrad minibatch size (ideal: 25 minibatches \
                        per epoch). for provided datasets, x for history and y for lit', type=int,\
                        default=150)
    parser.add_argument('-ep', '--num_epochs', help='number of training epochs, can also determine \
                         dynamically via validate method', type=int, default=61)
    parser.add_argument('-agr', '--adagrad_reset', help='reset sum of squared gradients after this many\
                         epochs', type=int, default=10)
    parser.add_argument('-lr', help='adagrad learning rate', type=float, default=0.01)
    parser.add_argument('-v', '--do_val', help='check performance on dev set after this many\
                         epochs', type=int, default=50)
    parser.add_argument('-o', '--output', help='desired location of output model', \
                         default='data/deep/params')

    args = vars(parser.parse_args())
    d = args['d']

    # load data
    train = pickle.load(open('data/deep/train', 'rb'))
    dev = pickle.load(open('data/deep/dev', 'rb'))
    vocab, vdict = pickle.load(open('data/deep/vocab', 'rb'))

    train_qs = train
    val_qs = dev
    print('total questions: ', len(train_qs))
    total = 0
    for qs, ans in train_qs:
        total += len(qs)
    print('total sentences: ', total)

    orig_We = pickle.load(open(args['We'], 'rb'))

    len_voc = orig_We.shape[1]
    print('vocab length: ', len_voc, ' We shape: ', orig_We.shape)

    # output log and parameter file destinations
    param_file = args['output']
    log_file = param_file.split('_')[0] + '_log'

    # generate params / We
    params = init_params(d, deep=3)

    # add We matrix to params
    params += (orig_We, )
    r = roll_params(params)

    dim = r.shape[0]
    print('parameter vector dimensionality:', dim)

    log = open(log_file, 'w')

    # minibatch adagrad training
    ag = Adagrad(r.shape, args['lr'])
    min_error = float('inf')

    print('step 1 of 2: training DAN (takes 2-3 hours)')
    for epoch in range(0, args['num_epochs']):

        lstring = ''

        # create mini-batches
        random.shuffle(train_qs)
        batches = [train_qs[x : x + args['batch_size']] for x in list(range(0, len(train_qs)),
                   args['batch_size'])]

        epoch_error = 0.0
        ep_t = time.time()

        for batch_ind, batch in enumerate(batches):
            now = time.time()
            err, grad = objective_and_grad(batch, r, d, len_voc)
            update = ag.rescale_update(grad)
            r -= update
            lstring = 'epoch: ' + str(epoch) + ' batch_ind: ' + str(batch_ind) + \
                    ' error, ' + str(err) + ' time = '+ str(time.time()-now) + ' sec'
            print(lstring)
            log.write(lstring + '\n')
            log.flush()

            epoch_error += err

        # done with epoch
        print(time.time() - ep_t)
        print('done with epoch ', epoch, ' epoch error = ', epoch_error, ' min error = ', min_error)
        lstring = 'done with epoch ' + str(epoch) + ' epoch error = ' + str(epoch_error) \
                 + ' min error = ' + str(min_error) + '\n\n'
        log.write(lstring)
        log.flush()

        # save parameters if the current model is better than previous best model
        if epoch_error < min_error:
            min_error = epoch_error
            print('saving model...')
            params = unroll_params(r, d, len_voc, deep=3)
            pickle.dump(params, open(param_file, 'wb'))

        # reset adagrad weights
        if epoch % args['adagrad_reset'] == 0 and epoch != 0:
            ag.reset_weights()

        # check accuracy on validation set
        if epoch % args['do_val'] == 0 and epoch != 0:
            print('validating...')
            params = unroll_params(r, d, len_voc, deep=3)
            evaluate(train_qs, val_qs, params, d)
            print('\n\n')

    log.close()

    print('step 2 of 2: training classifier over all answers (takes 10-15 hours depending on number of answers)')
    params = pickle.load(open(param_file, 'rb'))
    evaluate(train_qs, val_qs, params, d)
