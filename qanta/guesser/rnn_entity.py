from typing import List, Optional, Tuple
import time
import pickle
import os
import shutil
import re
import string

import numpy as np

import spacy
from spacy.tokens import Token

from sklearn.model_selection import train_test_split

import progressbar
import pycountry

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import Adam, lr_scheduler

from qanta import logging
from qanta.guesser.abstract import AbstractGuesser
from qanta.datasets.abstract import TrainingData, QuestionText
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.guesser.nn import compute_n_classes, create_embeddings
from qanta.torch import (
    BaseLogger, TerminateOnNaN, Tensorboard, create_save_model,
    EarlyStopping, ModelCheckpoint, MaxEpochStopping, TrainingManager
)
from qanta.util.io import safe_open


log = logging.get(__name__)

PT_RNN_ENTITY_WE_TMP = '/tmp/qanta/deep/pt_rnn_entity_we.pickle'
PT_RNN_ENTITY_WE = 'pt_rnn_entity_we.pickle'
UNK = 'UNK'
CUDA = torch.cuda.is_available()

LOWER_TO_UPPER = dict(zip(string.ascii_lowercase, string.ascii_uppercase))


def capitalize(sentence):
    if len(sentence) > 0:
        if sentence[0] in LOWER_TO_UPPER:
            return LOWER_TO_UPPER[sentence[0]] + sentence[1:]
        else:
            return sentence
    else:
        return sentence


qb_patterns = {
    '\n',
    ', for 10 points,',
    ', for ten points,',
    '--for 10 points--',
    'for 10 points, ',
    'for 10 points--',
    'for ten points, ',
    'for 10 points ',
    'for ten points ',
    ', ftp,'
    'ftp,',
    'ftp',
    '(*)'
}
re_pattern = '|'.join([re.escape(p) for p in qb_patterns])
re_pattern += r'|\[.*?\]|\(.*?\)'

COUNTRIES = set(['american'])
for c in pycountry.countries:
    split = c.name.lower().split()
    if len(split) == 1:
        COUNTRIES.add(split[0])

STATES = set()
for s in pycountry.subdivisions.get(country_code='US'):
    if s.type == 'State':
        split = s.name.lower().split()
        if len(split) == 1:
            STATES.add(split[0])

NN_TAGS = {'NN', 'NNP', 'NNS'}
SKIP_PUNCTATION = {'HYPH', 'POS'}
SKIP_TAGS = NN_TAGS | SKIP_PUNCTATION


def extract_mentions(tokens):
    begin = None
    mention_spans = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.lower_ == 'this' or t.lower_ == 'these':
            begin = i
        elif begin is not None and t.tag_ in NN_TAGS:
            if i + 1 < len(tokens) and tokens[i + 1].tag_ in SKIP_TAGS:
                i += 1
                continue
            mention_spans.append((begin, i))
            begin = None
        elif begin is not None and (t.lower_ in COUNTRIES or t.lower_ in STATES):
            if i + 1 < len(tokens) and (tokens[i + 1].tag_ in SKIP_TAGS):
                i += 1
                continue
            mention_spans.append((begin, i))
            begin = None
        i += 1
    return mention_spans


def mentions_to_sequence(mention_spans, tokens):
    m_sequence = [0] * len(tokens)
    for m_span in mention_spans:
        i = m_span[0]
        while i <= m_span[1]:
            m_sequence[i] = 1
            i += 1
    return m_sequence


def clean_sentence(sent):
    return capitalize(re.sub(re_pattern, '', sent.strip(), flags=re.IGNORECASE))


def custom_spacy_pipeline(nlp):
    return nlp.tagger, nlp.entity


class MultiVocab:
    def __init__(self, word_vocab=None, pos_vocab=None, iob_vocab=None, ent_type_vocab=None):
        if word_vocab is None:
            self.word = set()
        else:
            self.word = word_vocab

        if pos_vocab is None:
            self.pos = set()
        else:
            self.pos = pos_vocab

        if iob_vocab is None:
            self.iob = set()
        else:
            self.iob = iob_vocab

        if ent_type_vocab is None:
            self.ent_type = set()
        else:
            self.ent_type = ent_type_vocab



def preprocess_dataset(nlp, data: TrainingData, train_size=.9, vocab=None, class_to_i=None, i_to_class=None):
    classes = set(data[1])
    if class_to_i is None or i_to_class is None:
        class_to_i = {}
        i_to_class = []
        for i, ans_class in enumerate(classes):
            class_to_i[ans_class] = i
            i_to_class.append(ans_class)

    y_train = []
    y_test = []
    if vocab is None:
        vocab = MultiVocab()

    questions_with_answer = list(zip(data[0], data[1]))
    if train_size != 1:
        train, test = train_test_split(questions_with_answer, train_size=train_size)
    else:
        train = questions_with_answer
        test = []

    raw_x_train = []
    for q, ans in train:
        for sentence in q:
            raw_x_train.append(sentence)
            y_train.append(class_to_i[ans])

    log.info('Spacy is processing train input')
    bar = progressbar.ProgressBar()
    x_train = []
    for x in bar(raw_x_train):
        x_train.append(nlp(x))
    for doc in x_train:
        for word in doc:
            vocab.word.add(word.lower_)
            vocab.pos.add(word.pos_)
            vocab.iob.add(word.ent_iob_)
            vocab.ent_type.add(word.ent_type_)

    raw_x_test = []
    for q, ans in test:
        for sentence in q:
            raw_x_test.append(sentence)
            y_test.append(class_to_i[ans])

    log.info('Spacy is processing test input')
    bar = progressbar.ProgressBar()
    x_test = []
    for x in bar(raw_x_test):
        x_test.append(nlp(x))

    return x_train, y_train, x_test, y_test, vocab, class_to_i, i_to_class


class MultiEmbeddingLookup:
    def __init__(self, word_lookup, pos_lookup, iob_lookup, ent_type_lookup):
        self.word = word_lookup
        self.pos = pos_lookup
        self.iob = iob_lookup
        self.ent_type = ent_type_lookup



def convert_tokens_to_representations(tokens: List[Token], embedding_lookup: MultiEmbeddingLookup):
    w_indices = []
    pos_indices = []
    iob_indices = []
    ent_type_indices = []

    for t in tokens:
        if t.lower_ in embedding_lookup.word:
            w_indices.append(embedding_lookup.word[t.lower_])
        else:
            w_indices.append(embedding_lookup.word[UNK])

        if t.tag_ in embedding_lookup.pos:
            pos_indices.append(embedding_lookup.pos[t.tag_])
        else:
            pos_indices.append(embedding_lookup.pos[UNK])

        if t.ent_iob_ in embedding_lookup.iob:
            iob_indices.append(embedding_lookup.iob[t.ent_iob_])
        else:
            iob_indices.append(embedding_lookup.iob[UNK])

        if t.ent_type_ in embedding_lookup.ent_type:
            ent_type_indices.append(embedding_lookup.ent_type[t.ent_type_])
        else:
            ent_type_indices.append(embedding_lookup.ent_type[UNK])

    return w_indices, pos_indices, iob_indices, ent_type_indices


def load_multi_embeddings(
        multi_vocab: Optional[MultiVocab]=None, root_directory='') -> Tuple[np.ndarray, MultiEmbeddingLookup]:
    if os.path.exists(PT_RNN_ENTITY_WE_TMP):
        log.info('Loading embeddings from tmp cache')
        with safe_open(PT_RNN_ENTITY_WE_TMP, 'rb') as f:
            return pickle.load(f)
    elif os.path.exists(os.path.join(root_directory, PT_RNN_ENTITY_WE)):
        log.info('Loading embeddings from restored cache')
        with safe_open(os.path.join(root_directory, PT_RNN_ENTITY_WE), 'rb') as f:
            return pickle.load(f)
    else:
        if multi_vocab is None:
            raise ValueError('To create new embeddings a vocab is needed')
        with safe_open(PT_RNN_ENTITY_WE_TMP, 'wb') as f:
            log.info('Creating embeddings and saving to cache')
            word_embeddings, word_lookup = create_embeddings(multi_vocab.word, expand_glove=True, mask_zero=True)

            pos_lookup = {
                'MASK': 0,
                UNK: 1
            }
            for i, term in enumerate(multi_vocab.pos, start=2):
                pos_lookup[term] = i

            iob_lookup = {
                'MASK': 0,
                UNK: 1
            }
            for i, term in enumerate(multi_vocab.iob, start=2):
                iob_lookup[term] = i

            ent_type_lookup = {
                'MASK': 0,
                UNK: 1
            }
            for i, term in enumerate(multi_vocab.ent_type, start=2):
                ent_type_lookup[term] = i

            multi_embedding_lookup = MultiEmbeddingLookup(word_lookup, pos_lookup, iob_lookup, ent_type_lookup)
            combined = word_embeddings, multi_embedding_lookup
            pickle.dump(combined, f)
            return combined


def repackage_hidden(hidden, reset=False):
    if type(hidden) == Variable:
        if reset:
            return Variable(hidden.data.zero_())
        else:
            return Variable(hidden.data)
    else:
        return tuple(repackage_hidden(v, reset=reset) for v in hidden)


def pad_batch(x_batch, max_length):
    x_batch_padded = []
    for r in x_batch:
        pad_r = list(r)
        while len(pad_r) < max_length:
            pad_r.append(0)
        x_batch_padded.append(pad_r)
    return torch.from_numpy(np.array(x_batch_padded)).long()


def create_batch(x_array_w, x_array_pos, x_array_iob, x_array_type, x_array_mention, y_array):
    lengths = np.array([len(r) for r in x_array_w])
    max_length = np.max(lengths)
    length_sort = np.argsort(-lengths)
    x_w_batch = x_array_w[length_sort]
    x_pos_batch = x_array_pos[length_sort]
    x_iob_batch = x_array_iob[length_sort]
    x_type_batch = x_array_type[length_sort]
    x_mention_batch = x_array_mention[length_sort]

    y_batch = y_array[length_sort]
    lengths = lengths[length_sort]

    x_batch_w_padded = pad_batch(x_w_batch, max_length)
    x_batch_pos_padded = pad_batch(x_pos_batch, max_length)
    x_batch_iob_padded = pad_batch(x_iob_batch, max_length)
    x_batch_type_padded = pad_batch(x_type_batch, max_length)

    # Since this is a literal value, it needs to match the embedding data type as well as having an extra dimension
    # that is put in place by the embeddings for the above input
    x_batch_mention_padded = pad_batch(x_mention_batch, max_length).float().unsqueeze(2)
    y_batch = torch.from_numpy(y_batch).long()

    if CUDA:
        x_batch_w_padded = x_batch_w_padded.cuda()
        x_batch_pos_padded = x_batch_pos_padded.cuda()
        x_batch_iob_padded = x_batch_iob_padded.cuda()
        x_batch_type_padded = x_batch_type_padded.cuda()
        x_batch_mention_padded = x_batch_mention_padded.cuda()
        y_batch = y_batch.cuda()

    return (x_batch_w_padded, x_batch_pos_padded, x_batch_iob_padded, x_batch_type_padded, x_batch_mention_padded,
            lengths, y_batch, length_sort)


class BatchedDataset:
    def __init__(self, batch_size, multi_embedding_lookup, x_tokens, y_array, truncate=True, shuffle=True):
        self.x_array_w = []
        self.x_array_pos = []
        self.x_array_iob = []
        self.x_array_ent_type = []
        self.x_array_mention = []
        self.y_array = []

        for q in x_tokens:
            w_indicies, pos_indices, iob_indices, ent_type_indices = convert_tokens_to_representations(
                q, multi_embedding_lookup
            )
            mention_spans = extract_mentions(q)
            mention_flags = mentions_to_sequence(mention_spans, q)
            self.x_array_w.append(w_indicies)
            self.x_array_pos.append(pos_indices)
            self.x_array_iob.append(iob_indices)
            self.x_array_ent_type.append(ent_type_indices)
            self.x_array_mention.append(mention_flags)

        for i in range(len(x_tokens)):
            if len(self.x_array_w[i]) == 0:
                self.x_array_w[i].append(multi_embedding_lookup.word[UNK])

            if len(self.x_array_pos[i]) == 0:
                self.x_array_pos[i].append(multi_embedding_lookup.pos[UNK])

            if len(self.x_array_iob[i]) == 0:
                self.x_array_iob[i].append(multi_embedding_lookup.iob[UNK])

            if len(self.x_array_ent_type[i]) == 0:
                self.x_array_ent_type[i].append(multi_embedding_lookup.ent_type[UNK])

            if len(self.x_array_mention[i]) == 0:
                self.x_array_mention[i].append(0)

        self.x_array_w = np.array(self.x_array_w)
        self.x_array_pos = np.array(self.x_array_pos)
        self.x_array_iob = np.array(self.x_array_iob)
        self.x_array_ent_type = np.array(self.x_array_ent_type)
        self.x_array_mention = np.array(self.x_array_mention)
        self.y_array = np.array(y_array)

        self.n_examples = self.y_array.shape[0]

        self.t_x_w_batches = None
        self.t_x_pos_batches = None
        self.t_x_iob_batches = None
        self.t_x_type_batches = None
        self.t_x_mention_batches = None
        self.length_batches = None
        self.t_y_batches = None
        self.sort_batches = None
        self.n_batches = None
        self._batchify(batch_size, truncate=truncate, shuffle=shuffle)

    def _batchify(self, batch_size, truncate=True, shuffle=True):
        """
        Batches the stored dataset, and stores it
        :param batch_size:
        :param truncate:
        :param shuffle:
        :return:
        """
        self.n_batches = self.n_examples // batch_size
        if shuffle:
            random_order = np.random.permutation(self.n_examples)
            self.x_array_w = self.x_array_w[random_order]
            self.x_array_pos = self.x_array_pos[random_order]
            self.x_array_iob = self.x_array_iob[random_order]
            self.x_array_ent_type = self.x_array_ent_type[random_order]
            self.x_array_mention = self.x_array_mention[random_order]
            self.y_array = self.y_array[random_order]

        t_x_w_batches = []
        t_x_pos_batches = []
        t_x_iob_batches = []
        t_x_type_batches = []
        t_x_mention_batches = []
        length_batches = []
        t_y_batches = []
        sort_batches = []

        for b in range(self.n_batches):
            x_w_batch = self.x_array_w[b * batch_size:(b + 1) * batch_size]
            x_pos_batch = self.x_array_pos[b * batch_size:(b + 1) * batch_size]
            x_iob_batch = self.x_array_iob[b * batch_size:(b + 1) * batch_size]
            x_type_batch = self.x_array_ent_type[b * batch_size:(b + 1) * batch_size]
            x_mention_batch = self.x_array_mention[b * batch_size:(b + 1) * batch_size]
            y_batch = self.y_array[b * batch_size:(b + 1) * batch_size]
            x_w_batch, x_pos_batch, x_iob_batch, x_type_batch, x_mention_batch, lengths, y_batch, sort = create_batch(
                x_w_batch, x_pos_batch, x_iob_batch, x_type_batch, x_mention_batch,
                y_batch
            )

            t_x_w_batches.append(x_w_batch)
            t_x_pos_batches.append(x_pos_batch)
            t_x_iob_batches.append(x_iob_batch)
            t_x_type_batches.append(x_type_batch)
            t_x_mention_batches.append(x_mention_batch)
            length_batches.append(lengths)
            t_y_batches.append(y_batch)
            sort_batches.append(sort)

        if (not truncate) and (batch_size * self.n_batches < self.n_examples):
            x_w_batch = self.x_array_w[self.n_batches * batch_size:]
            x_pos_batch = self.x_array_pos[self.n_batches * batch_size:]
            x_iob_batch = self.x_array_iob[self.n_batches * batch_size:]
            x_type_batch = self.x_array_ent_type[self.n_batches * batch_size:]
            x_mention_batch = self.x_array_mention[self.n_batches * batch_size:]
            y_batch = self.y_array[self.n_batches * batch_size:]

            x_w_batch, x_pos_batch, x_iob_batch, x_type_batch, x_mention_batch, lengths, y_batch, sort = create_batch(
                x_w_batch, x_pos_batch, x_iob_batch, x_type_batch, x_mention_batch,
                y_batch
            )

            t_x_w_batches.append(x_w_batch)
            t_x_pos_batches.append(x_pos_batch)
            t_x_iob_batches.append(x_iob_batch)
            t_x_type_batches.append(x_type_batch)
            t_x_mention_batches.append(x_mention_batch)
            length_batches.append(lengths)
            t_y_batches.append(y_batch)
            sort_batches.append(sort)

        self.t_x_w_batches = t_x_w_batches
        self.t_x_pos_batches = t_x_pos_batches
        self.t_x_iob_batches = t_x_iob_batches
        self.t_x_type_batches = t_x_type_batches
        self.t_x_mention_batches = t_x_mention_batches
        self.length_batches = length_batches
        self.t_y_batches = t_y_batches
        self.sort_batches = sort_batches


class RnnEntityGuesser(AbstractGuesser):
    def __init__(self, max_epochs=100, batch_size=256, learning_rate=.001, max_grad_norm=5):
        super(RnnEntityGuesser, self).__init__()
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.class_to_i = None
        self.i_to_class = None
        self.vocab = None
        self.word_embeddings = None
        self.multi_embedding_lookup = None
        self.n_classes = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.nlp = None

    def guess(self,
              questions: List[QuestionText],
              max_n_guesses: Optional[int]):
        x_test_tokens = [self.nlp(x) for x in questions]
        y_test = np.zeros(len(questions))
        dataset = BatchedDataset(
            self.batch_size, self.multi_embedding_lookup, x_test_tokens, y_test,
            truncate=False, shuffle=False
        )

        self.model.eval()
        self.model.cuda()
        guesses = []
        hidden = self.model.init_hidden(self.batch_size)
        for b in range(len(dataset.t_x_w_batches)):
            t_x_w_batch = Variable(dataset.t_x_w_batches[b], volatile=True)
            t_x_pos_batch = Variable(dataset.t_x_pos_batches[b], volatile=True)
            t_x_iob_batch = Variable(dataset.t_x_iob_batches[b], volatile=True)
            t_x_type_batch = Variable(dataset.t_x_type_batches[b], volatile=True)
            t_x_mention_batch = Variable(dataset.t_x_mention_batches[b], volatile=True)

            length_batch = dataset.length_batches[b]
            sort_batch = dataset.sort_batches[b]

            if len(length_batch) != self.batch_size:
                # This could happen for the last batch which is shorter than batch_size
                hidden = self.model.init_hidden(len(length_batch))
            else:
                hidden = repackage_hidden(hidden, reset=True)

            out, hidden = self.model(
                t_x_w_batch, t_x_pos_batch, t_x_iob_batch, t_x_type_batch, t_x_mention_batch,
                length_batch, hidden
            )
            probs = F.softmax(out)
            scores, preds = torch.max(probs, 1)
            scores = scores.data.cpu().numpy()[np.argsort(sort_batch)]
            preds = preds.data.cpu().numpy()[np.argsort(sort_batch)]
            for p, s in zip(preds, scores):
                guesses.append([(self.i_to_class[p], s)])

        return guesses

    def train(self, training_data: TrainingData):
        log.info('Preprocessing the dataset')
        self.nlp = spacy.load('en', create_pipeline=custom_spacy_pipeline)
        x_train_tokens, y_train, x_test_tokens, y_test, vocab, class_to_i, i_to_class = preprocess_dataset(
            self.nlp, training_data
        )

        self.class_to_i = class_to_i
        self.i_to_class = i_to_class
        self.vocab = vocab

        log.info('Loading word embeddings')
        word_embeddings, multi_embedding_lookup = load_multi_embeddings(multi_vocab=vocab)
        self.word_embeddings = word_embeddings
        self.multi_embedding_lookup = multi_embedding_lookup


        log.info('Batching the dataset')
        train_dataset = BatchedDataset(self.batch_size, multi_embedding_lookup, x_train_tokens, y_train)
        test_dataset = BatchedDataset(self.batch_size, multi_embedding_lookup, x_test_tokens, y_test)
        self.n_classes = compute_n_classes(training_data[1])

        log.info('Initializing neural model')
        self.model = RnnEntityModel(
            len(multi_embedding_lookup.word),
            len(multi_embedding_lookup.pos),
            len(multi_embedding_lookup.iob),
            len(multi_embedding_lookup.ent_type),
            self.n_classes
        )
        self.model.init_weights(word_embeddings=word_embeddings)
        self.model.cuda()
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=5, verbose=True)

        manager = TrainingManager([
            BaseLogger(log_func=log.info), TerminateOnNaN(),
            EarlyStopping(monitor='test_acc', patience=10, verbose=1), MaxEpochStopping(100),
            ModelCheckpoint(create_save_model(self.model), '/tmp/rnn_entity.pt', monitor='test_acc')
            #Tensorboard('rnn_entity', log_dir='tb-logs')
        ])

        log.info('Starting training...')
        while True:
            self.model.train()
            train_acc, train_loss, train_time = self.run_epoch(train_dataset, evaluate=False)

            self.model.eval()
            test_acc, test_loss, test_time = self.run_epoch(test_dataset, evaluate=True)

            stop_training, reasons = manager.instruct(
                train_time, train_loss, train_acc,
                test_time, test_loss, test_acc
            )

            if stop_training:
                log.info(' '.join(reasons))
                break
            else:
                self.scheduler.step(test_acc)

        log.info('Done training')

    def run_epoch(self, batched_dataset: BatchedDataset, evaluate=False):
        if evaluate:
            batch_order = range(batched_dataset.n_batches)
        else:
            batch_order = np.random.permutation(batched_dataset.n_batches)

        batch_accuracies = []
        batch_losses = []
        epoch_start = time.time()
        hidden = self.model.init_hidden(self.batch_size)
        for batch in batch_order:
            t_x_w_batch = Variable(batched_dataset.t_x_w_batches[batch], volatile=evaluate)
            t_x_pos_batch = Variable(batched_dataset.t_x_pos_batches[batch], volatile=evaluate)
            t_x_iob_batch = Variable(batched_dataset.t_x_iob_batches[batch], volatile=evaluate)
            t_x_type_batch = Variable(batched_dataset.t_x_type_batches[batch], volatile=evaluate)
            t_x_mention_batch = Variable(batched_dataset.t_x_mention_batches[batch], volatile=evaluate)
            length_batch = batched_dataset.length_batches[batch]
            t_y_batch = Variable(batched_dataset.t_y_batches[batch], volatile=evaluate)

            self.model.zero_grad()
            hidden = repackage_hidden(hidden, reset=True)
            out, hidden = self.model(
                t_x_w_batch, t_x_pos_batch, t_x_iob_batch, t_x_type_batch, t_x_mention_batch,
                length_batch, hidden
            )
            _, preds = torch.max(out, 1)
            accuracy = torch.mean(torch.eq(preds, t_y_batch).float()).data[0]
            batch_loss = self.criterion(out, t_y_batch)
            if not evaluate:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            batch_accuracies.append(accuracy)
            batch_losses.append(batch_loss.data[0])

        epoch_end = time.time()

        return np.mean(batch_accuracies), np.mean(batch_losses), epoch_end - epoch_start

    def save(self, directory: str):
        shutil.copyfile('/tmp/rnn_entity.pt', os.path.join(directory, 'rnn_entity.pt'))
        with open(os.path.join(directory, 'rnn_entity.pickle'), 'wb') as f:
            pickle.dump({
                'vocab': self.vocab,
                'class_to_i': self.class_to_i,
                'i_to_class': self.i_to_class,
                'word_embeddings': self.word_embeddings,
                'multi_embedding_lookup': self.multi_embedding_lookup,
                'n_classes': self.n_classes,
                'max_epochs': self.max_epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'max_grad_norm': self.max_grad_norm
            }, f)

    @classmethod
    def load(cls, directory: str):
        with open(os.path.join(directory, 'rnn_entity.pickle'), 'rb') as f:
            params = pickle.load(f)

        guesser = RnnEntityGuesser()
        guesser.vocab = params['vocab']
        guesser.class_to_i = params['class_to_i']
        guesser.i_to_class = params['i_to_class']
        guesser.word_embeddings = params['word_embeddings']
        guesser.multi_embedding_lookup = params['multi_embedding_lookup']
        guesser.n_classes = params['n_classes']
        guesser.max_epochs = params['max_epochs']
        guesser.batch_size = params['batch_size']
        guesser.learning_rate = params['learning_rate']
        guesser.max_grad_norm = params['max_grad_norm']
        guesser.model = torch.load(os.path.join(directory, 'rnn_entity.pt'))
        guesser.nlp = spacy.load('en', create_pipeline=custom_spacy_pipeline)
        return  guesser

    @classmethod
    def targets(cls):
        return ['rnn_entity.pickle', 'rnn_entity.pt']

    def qb_dataset(self):
        return QuizBowlDataset(guesser_train=True)


class RnnEntityModel(nn.Module):
    def __init__(self, word_vocab_size, pos_vocab_size, iob_vocab_size, type_vocab_size,
                 n_classes, embedding_dim=300, dropout_prob=.3, recurrent_dropout_prob=.3,
                 n_hidden_layers=1, n_hidden_units=1000, bidirectional=True, rnn_type='lstm',
                 rnn_output='last_hidden'):
        super(RnnEntityModel, self).__init__()
        self.word_vocab_size = word_vocab_size
        self.pos_vocab_size = pos_vocab_size
        self.iob_vocab_size = iob_vocab_size
        self.type_vocab_size = type_vocab_size
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout_prob
        self.recurrent_dropout_prob = recurrent_dropout_prob
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_units = n_hidden_units
        self.bidirectional = bidirectional
        self.rnn_output = rnn_output

        self.dropout = nn.Dropout(dropout_prob)

        self.word_embeddings = nn.Embedding(word_vocab_size, embedding_dim, padding_idx=0)
        self.pos_embeddings = nn.Embedding(pos_vocab_size, 50, padding_idx=0)
        self.iob_embeddings = nn.Embedding(iob_vocab_size, 50, padding_idx=0)
        self.type_embeddings = nn.Embedding(type_vocab_size, 50, padding_idx=0)

        self.rnn_type = rnn_type
        if rnn_type == 'lstm':
            rnn_layer = nn.LSTM
        elif rnn_type == 'gru':
            rnn_layer = nn.GRU
        else:
            raise ValueError('Unrecognized rnn layer type')
        self.rnn = rnn_layer(embedding_dim + 150, n_hidden_units, n_hidden_layers,
                           dropout=recurrent_dropout_prob, batch_first=True, bidirectional=bidirectional)
        self.num_directions = int(bidirectional) + 1
        self.classification_layer = nn.Sequential(
            nn.Linear(n_hidden_units * self.num_directions * self.n_hidden_layers, n_classes),
            nn.BatchNorm1d(n_classes),
            nn.Dropout(dropout_prob)
        )

    def init_weights(self, word_embeddings=None):
        if word_embeddings is not None:
            self.word_embeddings.weight = nn.Parameter(torch.from_numpy(word_embeddings).float())

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.rnn_type == 'lstm':
            return (
                Variable(weight.new(self.n_hidden_layers * self.num_directions, batch_size, self.n_hidden_units).zero_()),
                Variable(weight.new(self.n_hidden_layers * self.num_directions, batch_size, self.n_hidden_units).zero_())
            )
        else:
            return Variable(weight.new(self.n_hidden_layers * self.num_directions, batch_size, self.n_hidden_units).zero_())

    def forward(self, word_idxs, pos_idxs, iob_idxs, type_idxs: Variable, mention_flags, lengths, hidden):
        word_embeddings = self.word_embeddings(word_idxs)
        pos_embeddings = self.pos_embeddings(pos_idxs)
        iob_embeddings = self.iob_embeddings(iob_idxs)
        type_embeddings = self.type_embeddings(type_idxs)
        embeddings = self.dropout(torch.cat([word_embeddings, pos_embeddings, iob_embeddings, type_embeddings], 2))
        embeddings = torch.cat([embeddings, mention_flags], 2)

        packed_input = nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True)

        output, hidden = self.rnn(packed_input, hidden)
        if self.rnn_output == 'last_hidden':
            if type(hidden) == tuple:
                final_hidden = hidden[0]
            else:
                final_hidden = hidden

            h_reshaped = final_hidden.transpose(0, 1).contiguous().view(word_idxs.data.shape[0], -1)

            return self.classification_layer(h_reshaped), hidden
        elif self.rnn_output == 'max_pool':
            padded_output, padded_lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            actual_batch_size = word_idxs.data.shape[0]
            pooled = []
            for i in range(actual_batch_size):
                max_pooled = padded_output[i][:padded_lengths[i]].mean(0)
                #max_pooled = padded_output[i][:padded_lengths[i]].max(0)[0]
                pooled.append(max_pooled)
            pooled = torch.cat(pooled).view(actual_batch_size, -1)
            return self.classification_layer(pooled), hidden
        else:
            raise ValueError('Unrecognized rnn_output option')

