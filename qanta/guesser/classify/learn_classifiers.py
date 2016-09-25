import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

#from functional import seq

from qanta import logging
from qanta.guesser.util.functions import relu
#from qanta.extractors.deep import load_model
from qanta.util.constants import (DEEP_VOCAB_TARGET, DEEP_WE_TARGET, CLASS_LABEL_TARGET, DEEP_DAN_PARAMS_TARGET, EVAL_RES_TARGET, N_GUESSES, DEEP_DAN_CLASSIFIER_TARGET,
                                  DEEP_DAN_PARAMS_TARGET, DEEP_DEV_TARGET)
from qanta.util.io import safe_open
import lasagne
from theano import tensor as T
from qanta.guesser.dan_gpu import build_dan
log = logging.get(__name__)


def compute_recall_accuracy():
    d = 300
    with open(DEEP_DEV_TARGET, 'rb') as f:
        val_qs = pickle.load(f)
    with open(DEEP_DAN_PARAMS_TARGET, 'rb') as f:
        (W, b, W2, b2, W3, b3, L) = pickle.load(f)
    with open( DEEP_DAN_CLASSIFIER_TARGET, 'rb') as f:
        classifier = pickle.load(f)
        class_labels = classifier.classes_
    recall = 0
    accuracy = 0
    total = 0
    wrong = []
    for qs, ans in val_qs:
        ans = ans[0]
        prev_sum = np.zeros((d, 1))
        history = []
        sent_position = 0
        for dist in qs:

            sent = qs[dist]

            # input is average of all nouns in sentence
            # av = average(L[:, sent], axis=1).reshape((d, 1))
            history += sent
            prev_sum += np.sum(L[:, sent], axis=1).reshape((d, 1))
            if len(history) == 0:
                av = np.zeros((d, 1))
            else:
                av = prev_sum / len(history)

            # apply non-linearity
            p = relu(W.dot(av) + b)
            p2 = relu(W2.dot(p) + b2)
            p3 = relu(W3.dot(p2) + b3)

            curr_feats = p3.ravel().reshape(1, -1)

            if sent_position + 1 == len(qs):
                p_dist = classifier.predict_proba(curr_feats)
                #correct = seq(zip(p_dist[0], class_labels))\
                #    .sorted(reverse=True)\
                #    .take(N_GUESSES)\
                #    .exists(lambda s: ans == s[1])
                accuracy += int(class_labels[p_dist.argmax()] == ans)
                recall += correct
                if not correct:
                    wrong.append((qs, ans))
            sent_position += 1
        total += 1

    return recall / total, accuracy / total, total, wrong

def print_recall_dan_gpu_at_n(fold_target=DEEP_DEV_TARGET, results_target=EVAL_RES_TARGET, n_guesses=N_GUESSES, max_examples=None):
    recall_array, total = compute_recall_accuracy_dan_gpu_to_n(fold_target=fold_target, n_guesses=n_guesses, max_examples=max_examples)
    pickle.dump((recall_array, total), open(EVAL_RES_TARGET, 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)
    print("Total: %s examples" %total)
    for i, recall in enumerate(recall_array):
        print("Recall at %i: %f" %(i+1, recall))

def compute_recall_accuracy_dan_gpu_to_n(fold_target=DEEP_DEV_TARGET, vocab_target= DEEP_VOCAB_TARGET, class_labels_target= CLASS_LABEL_TARGET, deep_target= DEEP_DAN_PARAMS_TARGET , we_target=DEEP_WE_TARGET, n_guesses=N_GUESSES, max_examples=None):
    corr = 0
    wrong = []
    total = 0
    recall_at_n =  np.zeros(n_guesses,)
   #c1 = Counter()
    with open(fold_target, 'rb') as f:
        val_qs = pickle.load(f)
    #fit_fn, pred_fn, debug_fn, l_out  = load_model(deep_target)
    class_labels, rev_class_labels = pickle.load(open(class_labels_target, 'rb'), encoding='latin1')
    vocab, vdict = pickle.load(open(vocab_target, 'rb'), encoding='latin1')
    we = pickle.load(open(we_target, 'rb'))
    len_voc = len(vocab)
    num_labels = len(class_labels)
    sents = T.imatrix(name='sentence')
    masks = T.matrix(name='mask')
    labels = T.ivector('target')
    train_fn, pred_fn, debug_fn, l_out = build_dan(sents, masks, labels, len_voc, num_labels, We=we.T)
    params = pickle.load(open(deep_target, 'rb'), encoding='latin1')
    lasagne.layers.set_all_param_values(l_out, params)
    found_answers = np.array([vdict[x] for x in class_labels.keys()])
    for qs, ans in val_qs:
        ans = ans[0]
        history = []
        sent_position = 0
        for dist in qs:
            sent = qs[dist]
            history += sent
            p3 = np.array(history)
            curr_feats = p3.ravel().reshape(1,-1)
            mask = np.zeros((1, curr_feats.shape[1])).astype('float32')
            if sent_position + 1 == len(qs):
                p_dist = pred_fn(curr_feats.astype('int32'), mask)[0]
                p_dist_sorted = np.sort(p_dist)[::-1]
                if (np.where(found_answers == ans)[0].shape[0] > 0):
                   correct_prob = p_dist[np.where(found_answers == ans)[0][0]]
                   correct_index = np.where(p_dist_sorted == correct_prob)[0][0]
                   recall_at_n[correct_index:] += 1
                if not correct_index > 0:
                    wrong.append((qs, ans))
            sent_position += 1
        total += 1

    return recall_at_n / total,  total



def compute_recall_accuracy_to_n(fold_target=DEEP_DEV_TARGET, n_guesses=N_GUESSES, max_examples=None):
    d = 300
    with open(fold_target, 'rb') as f:
        val_qs = pickle.load(f)
    with open(DEEP_DAN_PARAMS_TARGET, 'rb') as f:
        (W, b, W2, b2, W3, b3, L) = pickle.load(f)
    with open( DEEP_DAN_CLASSIFIER_TARGET, 'rb') as f:
        classifier = pickle.load(f)
        class_labels = classifier.classes_
    recall_at_n = np.zeros(n_guesses,)
    total = 0
    wrong = []
    if max_examples:
       val_qs = val_qs[:max_examples]
    for qs, ans in val_qs:
        ans = ans[0]
        prev_sum = np.zeros((d, 1))
        history = []
        sent_position = 0
        for dist in qs:
            sent = qs[dist]
            history += sent
            prev_sum += np.sum(L[:, sent], axis=1).reshape((d, 1))
            if len(history) == 0:
                av = np.zeros((d, 1))
            else:
                av = prev_sum / len(history)

            # apply non-linearity
            p = relu(W.dot(av) + b)
            p2 = relu(W2.dot(p) + b2)
            p3 = relu(W3.dot(p2) + b3)

            curr_feats = p3.ravel().reshape(1,-1)

            if sent_position + 1 == len(qs):
                p_dist = classifier.predict_proba(curr_feats)
                p_dist_sorted = np.sort(p_dist[0])[::-1]
                if (np.where(class_labels == ans)[0].shape[0] > 0):
                   correct_prob = p_dist[0,np.where(class_labels == ans)[0][0]]
                   correct_index = np.where(p_dist_sorted == correct_prob)[0][0]
                   recall_at_n[correct_index:] += 1
            sent_position += 1
        total += 1

def print_recall_at_n(fold_target=DEEP_DEV_TARGET, results_target=EVAL_RES_TARGET, n_guesses=N_GUESSES, max_examples=None):
    recall_array, total = compute_recall_accuracy_to_n(fold_target=fold_target, n_guesses=n_guesses, max_examples=max_examples)
    pickle.dump((recall_array, total), open(EVAL_RES_TARGET, 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)
    print("Total: %s examples" %total)
    for i, recall in enumerate(recall_array):
        print("Recall at %i: %f" %(i+1, recall))


def compute_recall_accuracy_to_n(fold_target=DEEP_DEV_TARGET, n_guesses=N_GUESSES,
                                 max_examples=None):
    d = 300
    with open(fold_target, 'rb') as f:
        val_qs = pickle.load(f)
    with open(DEEP_DAN_PARAMS_TARGET, 'rb') as f:
        (W, b, W2, b2, W3, b3, L) = pickle.load(f)
    with open( DEEP_DAN_CLASSIFIER_TARGET, 'rb') as f:
        classifier = pickle.load(f)
        class_labels = classifier.classes_
    recall_at_n = np.zeros(n_guesses,)
    total = 0
    wrong = []
    if max_examples:
       val_qs = val_qs[:max_examples]
    for qs, ans in val_qs:
        ans = ans[0]
        prev_sum = np.zeros((d, 1))
        history = []
        sent_position = 0
        for dist in qs:
            sent = qs[dist]
            history += sent
            prev_sum += np.sum(L[:, sent], axis=1).reshape((d, 1))
            if len(history) == 0:
                av = np.zeros((d, 1))
            else:
                av = prev_sum / len(history)

            # apply non-linearity
            p = relu(W.dot(av) + b)
            p2 = relu(W2.dot(p) + b2)
            p3 = relu(W3.dot(p2) + b3)

            curr_feats = p3.ravel().reshape(1,-1)

            if sent_position + 1 == len(qs):
                p_dist = classifier.predict_proba(curr_feats)
                p_dist_sorted = np.sort(p_dist[0])[::-1]
                if (np.where(class_labels == ans)[0].shape[0] > 0):
                   correct_prob = p_dist[0,np.where(class_labels == ans)[0][0]]
                   correct_index = np.where(p_dist_sorted == correct_prob)[0][0]
                   recall_at_n[correct_index:] += 1
                if not correct_index > 0:
                    wrong.append((qs, ans))
            sent_position += 1
        total += 1

    return recall_at_n / total,  total


def print_recall_at_n(fold_target=DEEP_DEV_TARGET, results_target=EVAL_RES_TARGET, n_guesses=N_GUESSES, max_examples=None):
    recall_array, total = compute_recall_accuracy_to_n(fold_target=fold_target, n_guesses=n_guesses, max_examples=max_examples)
    pickle.dump((recall_array, total), open(EVAL_RES_TARGET, 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)
    print("Total: %s examples" %total)
    for i, recall in enumerate(recall_array):
        print("Recall at %i: %f" %(i+1, recall))


def compute_vectors(train_qs, test_qs, params, d):
    data = [train_qs, test_qs]
    (W, b, W2, b2, W3, b3, L) = params

    train_vector = []
    test_vector = []

    for tt, split in enumerate(data):
        for qs, ans in split:
            prev_sum = np.zeros((d, 1))
            history = []

            for dist in qs:

                sent = qs[dist]

                # input is average of all nouns in sentence
                # av = average(L[:, sent], axis=1).reshape((d, 1))
                history += sent
                prev_sum += np.sum(L[:, sent], axis=1).reshape((d, 1))
                if len(history) == 0:
                    av = np.zeros((d, 1))
                else:
                    av = prev_sum / len(history)

                # apply non-linearity
                p = relu(W.dot(av) + b)
                p2 = relu(W2.dot(p) + b2)
                p3 = relu(W3.dot(p2) + b3)

                if tt == 0:
                    train_vector.append((p3.ravel(), ans[0]))

                else:
                    test_vector.append((p3.ravel(), ans[0]))
    return train_vector, test_vector


# trains a classifier, saves it to disk, and evaluates on heldout data
def evaluate(train_vector, test_vector):
    log.info('total training instances: {0}'.format(len(train_vector[0])))
    log.info('total testing instances: {0}'.format(len(test_vector[0])))

    classifier = OneVsRestClassifier(LogisticRegression(C=10), n_jobs=-1)
    classifier.fit(train_vector[0], train_vector[1])

    with safe_open(DEEP_DAN_CLASSIFIER_TARGET, 'wb') as f:
        pickle.dump(classifier, f, protocol=pickle.HIGHEST_PROTOCOL)

    train_accuracy = classifier.score(X=train_vector[0], y=train_vector[1])
    test_accuracy = classifier.score(X=test_vector[0], y=test_vector[1])
    log.info('accuracy train: {0}'.format(train_accuracy))
    log.info('accuracy test: {0}'.format(test_accuracy))
