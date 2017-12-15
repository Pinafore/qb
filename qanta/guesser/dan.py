from typing import List, Tuple, Optional
from pprint import pformat
import shutil
import os
import pickle
import time

import numpy as np
import hyperopt as ho
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import Adam, SGD, lr_scheduler

from qanta import logging
from qanta.config import conf
from qanta.guesser.abstract import AbstractGuesser
from qanta.datasets.abstract import TrainingData, Answer, QuestionText
from qanta.datasets.wikipedia import WikipediaDataset
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.preprocess import preprocess_dataset, tokenize_question
from qanta.guesser.nn import create_load_embeddings_function, convert_text_to_embeddings_indices, compute_n_classes
from qanta.torch import (
    embedded_dropout, BaseLogger, TerminateOnNaN,
    EarlyStopping, ModelCheckpoint, MaxEpochStopping, TrainingManager
)


log = logging.get(__name__)


PTDAN_WE_TMP = '/tmp/qanta/deep/pt_dan_we.pickle'
PTDAN_WE = 'pt_dan_we.pickle'
load_embeddings = create_load_embeddings_function(PTDAN_WE_TMP, PTDAN_WE, log)
CUDA = torch.cuda.is_available()


def create_save_model(model):
    def save_model(path):
        torch.save(model.state_dict(), path)
    return save_model


def pad_batch(x_batch):
    x_lengths = np.array([len(r) for r in x_batch])
    if x_lengths.min() == 0:
        raise ValueError('Should not have zero length sequences')
    max_len = x_lengths.max()
    padded_x_batch = []
    for r in x_batch:
        pad_r = list(r)
        while len(pad_r) < max_len:
            pad_r.append(0)  # 0 is the mask idx
        padded_x_batch.append(pad_r)

    #offsets = np.cumsum([0] + x_lengths[:-1])
    return np.array(padded_x_batch), x_lengths


def batchify(batch_size, x_array, y_array, truncate=True, shuffle=True):
    n_examples = x_array.shape[0]
    n_batches = n_examples // batch_size
    if shuffle:
        random_order = np.random.permutation(n_examples)
        x_array = x_array[random_order]
        y_array = y_array[random_order]

    t_x_batches = []
    t_len_batches = []
    t_y_batches = []

    for b in range(n_batches):
        x_batch = x_array[b * batch_size:(b + 1) * batch_size]
        y_batch = y_array[b * batch_size:(b + 1) * batch_size]
        flat_x_batch, lens = pad_batch(x_batch)

        if CUDA:
            t_x_batches.append(torch.from_numpy(flat_x_batch).long().cuda())
            t_len_batches.append(torch.from_numpy(lens).float().cuda())
            t_y_batches.append(torch.from_numpy(y_batch).long().cuda())
        else:
            t_x_batches.append(torch.from_numpy(flat_x_batch).long())
            t_len_batches.append(torch.from_numpy(lens).float())
            t_y_batches.append(torch.from_numpy(y_batch).long())

    if (not truncate) and (batch_size * n_batches < n_examples):
        x_batch = x_array[n_batches * batch_size:]
        y_batch = y_array[n_batches * batch_size:]
        flat_x_batch, lens = pad_batch(x_batch)

        if CUDA:
            t_x_batches.append(torch.from_numpy(flat_x_batch).long().cuda())
            t_len_batches.append(torch.from_numpy(lens).float().cuda())
            t_y_batches.append(torch.from_numpy(y_batch).long().cuda())
        else:
            t_x_batches.append(torch.from_numpy(flat_x_batch).long())
            t_len_batches.append(torch.from_numpy(lens).float())
            t_y_batches.append(torch.from_numpy(y_batch).long())

    t_x_batches = np.array(t_x_batches, dtype=np.object)
    t_len_batches = np.array(t_len_batches, dtype=np.object)
    t_y_batches = np.array(t_y_batches, dtype=np.object)

    return n_batches, t_x_batches, t_len_batches, t_y_batches


class DanGuesser(AbstractGuesser):
    def __init__(self):
        super(DanGuesser, self).__init__()
        guesser_conf = conf['guessers']['Dan']
        self.use_wiki = guesser_conf['use_wiki']
        self.use_qb = guesser_conf['use_qb']
        self.use_buzz_as_train = conf['buzz_as_guesser_train']
        self.optimizer_name = guesser_conf['optimizer']
        self.sgd_weight_decay = guesser_conf['sgd_weight_decay']
        self.sgd_lr = guesser_conf['sgd_lr']
        self.adam_lr = guesser_conf['adam_lr']
        self.batch_size = guesser_conf['batch_size']
        self.max_epochs = guesser_conf['max_epochs']
        self.use_lr_scheduler = guesser_conf['use_lr_scheduler']
        self.gradient_clip = guesser_conf['gradient_clip']
        self.emb_dropout = guesser_conf['emb_dropout']
        self.sm_dropout = guesser_conf['sm_dropout']
        self.nn_dropout = guesser_conf['nn_dropout']
        self.hyper_opt = guesser_conf['hyper_opt']

        self.class_to_i = None
        self.i_to_class = None
        self.vocab = None
        self.embeddings = None
        self.embedding_lookup = None
        self.n_classes = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.vocab_size = None

    def parameters(self):
        return {
            'use_wiki': self.use_wiki,
            'use_qb': self.use_qb,
            'use_buzz_as_train': self.use_buzz_as_train,
            'optimizer_name': self.optimizer_name,
            'sgd_weight_decay': self.sgd_weight_decay,
            'sgd_lr': self.sgd_lr,
            'adam_lr': self.adam_lr,
            'batch_size': self.batch_size,
            'max_epochs': self.max_epochs,
            'use_lr_scheduler': self.use_lr_scheduler,
            'gradient_clip': self.gradient_clip,
            'emb_dropout': self.emb_dropout,
            'nn_dropout': self.nn_dropout,
            'sm_dropout': self.sm_dropout,
            'hyper_opt': self.hyper_opt
        }

    def guess(self, questions: List[QuestionText], max_n_guesses: Optional[int]) -> List[List[Tuple[Answer, float]]]:
        x_test = [convert_text_to_embeddings_indices(
            tokenize_question(q), self.embedding_lookup)
            for q in questions
        ]
        for r in x_test:
            if len(r) == 0:
                log.warn('Found an empty question, adding an UNK token to it so that NaNs do not occur')
                r.append(self.embedding_lookup['UNK'])
        x_test = np.array(x_test)
        y_test = np.zeros(len(x_test))

        _, t_x_batches, t_len_batches, t_y_batches = batchify(
            self.batch_size, x_test, y_test, truncate=False, shuffle=False
        )

        self.model.eval()
        if CUDA:
            self.model = self.model.cuda()

        guesses = []
        for b in range(len(t_x_batches)):
            t_x = Variable(t_x_batches[b], volatile=True)
            t_len = Variable(t_len_batches[b], volatile=True)
            out = self.model(t_x, t_len)
            probs = F.softmax(out)
            scores, preds = torch.max(probs, 1)
            scores = scores.data.cpu().numpy()
            preds = preds.data.cpu().numpy()
            for p, s in zip(preds, scores):
                guesses.append([(self.i_to_class[p], s)])

        return guesses

    def train(self, training_data: TrainingData) -> None:
        if self.use_qb:
            x_train_text, y_train, x_test_text, y_test, vocab, class_to_i, i_to_class = preprocess_dataset(
                training_data
            )
            if self.use_wiki:
                wiki_dataset = WikipediaDataset(set(training_data[1]))
                wiki_train_data = wiki_dataset.training_data()
                w_x_train_text, w_train_y, _, _, _, _, _ = preprocess_dataset(
                    wiki_train_data, train_size=1, vocab=vocab, class_to_i=class_to_i, i_to_class=i_to_class
                )
                x_train_text.extend(w_x_train_text)
                y_train.extend(w_train_y)
        else:
            if self.use_wiki:
                wiki_dataset = WikipediaDataset(set(training_data[1]))
                wiki_train_data = wiki_dataset.training_data()
                x_train_text, y_train, x_test_text, y_test, vocab, class_to_i, i_to_class = preprocess_dataset(
                    wiki_train_data
                )
            else:
                raise ValueError('use_wiki and use_qb cannot both be false, otherwise there is no training data')

        self.class_to_i = class_to_i
        self.i_to_class = i_to_class
        self.vocab = vocab

        embeddings, embedding_lookup = load_embeddings(vocab=vocab, expand_glove=True, mask_zero=True)
        self.embeddings = embeddings
        self.embedding_lookup = embedding_lookup

        x_train = [convert_text_to_embeddings_indices(q, embedding_lookup) for q in x_train_text]
        for r in x_train:
            if len(r) == 0:
                r.append(embedding_lookup['UNK'])
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        x_test = [convert_text_to_embeddings_indices(q, embedding_lookup) for q in x_test_text]
        for r in x_test:
            if len(r) == 0:
                r.append(embedding_lookup['UNK'])
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        self.n_classes = compute_n_classes(training_data[1])

        log.info(f'Batching: {len(x_train)} train questions and {len(x_test)} test questions')

        n_batches_train, t_x_train, t_len_train, t_y_train = batchify(
            self.batch_size, x_train, y_train, truncate=True)

        n_batches_test, t_x_test, t_len_test, t_y_test = batchify(
            self.batch_size, x_test, y_test, truncate=False)

        self.vocab_size = embeddings.shape[0]
        if self.hyper_opt:
            self.hyperparameter_optimize(
                embeddings,
                n_batches_train, t_x_train, t_len_train, t_y_train,
                n_batches_test, t_x_test, t_len_test, t_y_test
            )
        else:
            self._fit(
                embeddings,
                n_batches_train, t_x_train, t_len_train, t_y_train,
                n_batches_test, t_x_test, t_len_test, t_y_test
            )

    def hyperparameter_optimize(self,
                                embeddings, n_batches_train, t_x_train, t_len_train, t_y_train,
                                n_batches_test, t_x_test, t_len_test, t_y_test):
        scores = []
        params = []
        space = {
            'sm_dropout': ho.hp.uniform('sm_dropout', 0, 1),
            'nn_dropout': ho.hp.uniform('nn_dropuot', 0, 1),
            'emb_dropout': ho.hp.uniform('emb_dropout', 0, 1)

        }

        def objective(sampled_params):
            p = {
                'sm_dropout': sampled_params['sm_dropout'],
                'nn_dropout': sampled_params['nn_dropout'],
                'emb_dropout': sampled_params['emb_dropout']
            }
            acc_score = self._fit(
                embeddings,
                n_batches_train, t_x_train, t_len_train, t_y_train,
                n_batches_test, t_x_test, t_len_test, t_y_test,
                hyper_params=p
            )
            return {'loss': -acc_score, 'status': ho.STATUS_OK}

        trials = ho.Trials()
        best = ho.fmin(
            fn=objective, space=space, algo=ho.tpe.suggest, max_evals=100,
            trials=trials
        )
        print(best)
        print(trials)

        while True:
            sm_dropout = np.random.uniform()
            nn_dropout = np.random.uniform()
            embed_dropout = np.random.uniform()
            p = {
                'sm_dropout': sm_dropout,
                'nn_dropout': nn_dropout,
                'emb_dropout': embed_dropout
            }
            acc_score = self._fit(
                embeddings,
                n_batches_train, t_x_train, t_len_train, t_y_train,
                n_batches_test, t_x_test, t_len_test, t_y_test,
                hyper_params=p
            )
            scores.append(acc_score)
            params.append(p)
            log.info(f'Trial {len(scores)}')
            best = np.argmax(scores)
            log.info(f'Best score: {scores[best]} params: {params[best]}')
        log.info(f'Parameter sweep results: {results}')
        raise ValueError('Hyper parameter optimization done')

    def _fit(self, embeddings,
             n_batches_train, t_x_train, t_len_train, t_y_train,
             n_batches_test, t_x_test, t_len_test, t_y_test, hyper_params=None):
        model_params = {
            'emb_dropout': self.emb_dropout,
            'sm_dropout': self.sm_dropout,
            'nn_dropout': self.nn_dropout,
            'adam_lr': self.adam_lr,
            'sgd_lr': self.sgd_lr
        }
        if hyper_params is not None:
            for k, v in hyper_params.items():
                model_params[k] = v
        self.model = DanModel(
            self.vocab_size, self.n_classes,
            embeddings=embeddings,
            emb_dropout_prob=model_params['emb_dropout'],
            sm_dropout_prob=model_params['sm_dropout'],
            nn_dropout_prob=model_params['nn_dropout']
        )
        if CUDA:
            self.model = self.model.cuda()
        log.info(f'Model:\n{self.model}')
        log.info(f'Parameters:\n{pformat(self.parameters())}')
        if self.hyper_opt:
            log.info(f'Hyper params:\n{pformat(model_params)}')

        if self.optimizer_name == 'adam':
            lr = model_params['adam_lr']
            self.optimizer = Adam(self.model.parameters(), lr=lr)
        elif self.optimizer_name == 'sgd':
            lr = model_params['sgd_lr']
            self.optimizer = SGD(self.model.parameters(), lr=lr, weight_decay=self.sgd_weight_decay)
        else:
            raise ValueError('Invalid optimizer')
        self.criterion = nn.CrossEntropyLoss()

        if self.use_lr_scheduler:
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, verbose=True, mode='max')

        manager = TrainingManager([
            BaseLogger(log_func=log.info), TerminateOnNaN(),
            EarlyStopping(monitor='test_acc', patience=10, verbose=1), MaxEpochStopping(100),
            ModelCheckpoint(create_save_model(self.model), '/tmp/dan.pt', monitor='test_acc')
        ])

        log.info('Starting training...')
        best_acc = 0.0
        while True:
            self.model.train()
            train_acc, train_loss, train_time = self.run_epoch(
                n_batches_train,
                t_x_train, t_len_train, t_y_train, evaluate=False
            )

            self.model.eval()
            test_acc, test_loss, test_time = self.run_epoch(
                n_batches_test,
                t_x_test, t_len_test, t_y_test, evaluate=True
            )
            best_acc = max(best_acc, test_acc)

            stop_training, reasons = manager.instruct(
                train_time, train_loss, train_acc,
                test_time, test_loss, test_acc
            )

            if stop_training:
                log.info(' '.join(reasons))
                break
            else:
                if self.use_lr_scheduler:
                    self.scheduler.step(test_acc)

        log.info('Done training')
        return best_acc

    def run_epoch(self, n_batches, t_x_array, t_len_array, t_y_array, evaluate=False):
        if not evaluate:
            random_batch_order = np.random.permutation(n_batches)
            t_x_array = t_x_array[random_batch_order]
            t_len_array = t_len_array[random_batch_order]
            t_y_array = t_y_array[random_batch_order]

        batch_accuracies = []
        batch_losses = []
        epoch_start = time.time()
        for batch in range(n_batches):
            t_x_batch = Variable(t_x_array[batch], volatile=evaluate)
            t_len_batch = Variable(t_len_array[batch], volatile=evaluate)
            t_y_batch = Variable(t_y_array[batch], volatile=evaluate)

            self.model.zero_grad()
            out = self.model(t_x_batch, t_len_batch)
            _, preds = torch.max(out, 1)
            accuracy = torch.mean(torch.eq(preds, t_y_batch).float()).data[0]
            batch_loss = self.criterion(out, t_y_batch)
            if not evaluate:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()

            batch_accuracies.append(accuracy)
            batch_losses.append(batch_loss.data[0])

        epoch_end = time.time()

        return np.mean(batch_accuracies), np.mean(batch_losses), epoch_end - epoch_start

    def save(self, directory: str) -> None:
        shutil.copyfile('/tmp/dan.pt', os.path.join(directory, 'dan.pt'))
        with open(os.path.join(directory, 'dan.pickle'), 'wb') as f:
            pickle.dump({
                'vocab': self.vocab,
                'class_to_i': self.class_to_i,
                'i_to_class': self.i_to_class,
                'embeddings': self.embeddings,
                'embeddings_lookup': self.embedding_lookup,
                'n_classes': self.n_classes,
                'max_epochs': self.max_epochs,
                'batch_size': self.batch_size,
                'adam_lr': self.adam_lr,
                'sgd_lr': self.sgd_lr,
                'use_wiki': self.use_wiki,
                'use_qb': self.use_qb,
                'vocab_size': self.vocab_size,
                'emb_dropout': self.emb_dropout,
                'nn_dropout': self.nn_dropout,
                'sm_dropout': self.sm_dropout,
                'hyper_opt': self.hyper_opt
            }, f)

    @classmethod
    def load(cls, directory: str):
        with open(os.path.join(directory, 'dan.pickle'), 'rb') as f:
            params = pickle.load(f)

        guesser = DanGuesser()
        guesser.vocab = params['vocab']
        guesser.class_to_i = params['class_to_i']
        guesser.i_to_class = params['i_to_class']
        guesser.embeddings = params['embeddings']
        guesser.embedding_lookup = params['embeddings_lookup']
        guesser.n_classes = params['n_classes']
        guesser.max_epochs = params['max_epochs']
        guesser.batch_size = params['batch_size']
        guesser.use_wiki = params['use_wiki']
        guesser.adam_lr = params['adam_lr']
        guesser.sgd_lr = params['sgd_lr']
        guesser.use_qb = params['use_qb']
        guesser.vocab_size = params['vocab_size']
        guesser.emb_dropout = params['emb_dropout']
        guesser.nn_dropout = params['nn_dropout']
        guesser.sm_dropout = params['sm_dropout']
        guesser.hyper_opt = params['hyper_opt']
        guesser.model = DanModel(guesser.vocab_size, guesser.n_classes)
        guesser.model.load_state_dict(torch.load(
            os.path.join(directory, 'dan.pt'), map_location=lambda storage, loc: storage
        ))
        return guesser

    @classmethod
    def targets(cls) -> List[str]:
        return ['dan.pickle', 'dan.pt']

    def qb_dataset(self):
        return QuizBowlDataset(guesser_train=True, buzzer_train=self.use_buzz_as_train)


class DanModel(nn.Module):
    def __init__(self, vocab_size, n_classes,
                 embeddings=None,
                 embedding_dim=300, nn_dropout_prob=.3, sm_dropout_prob=.3, emb_dropout_prob=.05,
                 n_hidden_layers=1, n_hidden_units=1000, non_linearity='elu'):
        super(DanModel, self).__init__()
        self.n_hidden_layers = 1
        self.non_linearity = non_linearity
        if non_linearity == 'relu':
            self._non_linearity = nn.ReLU
        elif non_linearity == 'elu':
            self._non_linearity = nn.ELU
        elif non_linearity == 'prelu':
            self._non_linearity = nn.PReLU
        else:
            raise ValueError('Unrecognized non-linearity function:{}'.format(non_linearity))
        self.n_hidden_units = n_hidden_units
        self.nn_dropout_prob = nn_dropout_prob
        self.sm_dropout_prob = sm_dropout_prob
        self.emb_dropout_prob = emb_dropout_prob
        self.vocab_size = vocab_size
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim

        self.dropout = nn.Dropout(nn_dropout_prob)

        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embeddings is not None:
            self.embeddings.weight.data = torch.from_numpy(embeddings).float()

        layers = []
        for i in range(n_hidden_layers):
            if i == 0:
                input_dim = embedding_dim
            else:
                input_dim = n_hidden_units

            layers.extend([
                nn.Linear(input_dim, n_hidden_units),
                nn.BatchNorm1d(n_hidden_units),
                self._non_linearity(),
                nn.Dropout(nn_dropout_prob),
            ])

        layers.extend([
            nn.Linear(n_hidden_units, n_classes),
            nn.BatchNorm1d(n_classes),
            nn.Dropout(sm_dropout_prob)
        ])
        self.layers = nn.Sequential(*layers)

    def forward(self, input_: Variable, lengths: Variable):
        q_enc = embedded_dropout(self.embeddings, input_, dropout=self.emb_dropout_prob)
        q_enc = q_enc.sum(1) / lengths.view(input_.size()[0], -1)
        return self.layers(self.dropout(q_enc))
