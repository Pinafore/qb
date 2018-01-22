from typing import List, Optional, Tuple
import json
import time
import pickle
import os
import shutil
import re
import string
from pprint import pformat

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

from qanta import qlogging
from qanta.config import conf
from qanta.guesser.abstract import AbstractGuesser
from qanta.datasets.abstract import TrainingData, QuestionText
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.datasets.wikipedia import WikipediaDataset, TagmeWikipediaDataset
from qanta.datasets.triviaqa import TriviaQADataset
from qanta.guesser.nn import compute_n_classes, create_embeddings
from qanta.torch import (
    BaseLogger, TerminateOnNaN, Tensorboard, create_save_model,
    EarlyStopping, ModelCheckpoint, MaxEpochStopping, TrainingManager
)
from qanta.torch.nn import WeightDrop, LockedDropout
from qanta.util.io import safe_open


log = qlogging.get(__name__)

PT_RNN_ENTITY_WE_TMP = '/tmp/qanta/deep/pt_rnn_entity_we.pickle'
PT_RNN_ENTITY_WE = 'pt_rnn_entity_we.pickle'
UNK = 'UNK'
EOS = 'EOS'
WIKI_TITLE_MENTION = 'wikititlemention'
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

COUNTRIES = {'american'}
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
PRONOUNS = {'they', 'it', 'he', 'she'}
PREPOSITION_POS = {'ADP', 'ADV'}


def extract_this_mentions(tokens):
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


def extract_pronoun_mentions(doc):
    mentions = set()
    if len(doc) == 0:
        return mentions
    for sent in doc.sents:
        is_start_of_sentence = True
        is_phrase_span = False
        for t in sent:
            if is_start_of_sentence and not is_phrase_span and t.pos_ in PREPOSITION_POS:
                is_phrase_span = True
            if is_start_of_sentence and t.lower_ in PRONOUNS:
                mentions.add(t.i)
                is_start_of_sentence = False
            elif is_phrase_span and t.text == ',':
                is_phrase_span = False
                is_start_of_sentence = True
            else:
                is_start_of_sentence = False
    return mentions


def extract_title_mentions(doc):
    mention_spans = []
    if len(doc) == 0:
        return mention_spans

    for sent in doc.sents:
        for t in sent:
            if t.lower_ == WIKI_TITLE_MENTION:
                mention_spans.append((t.i, t.i))

    return mention_spans

def mentions_to_sequence(mention_spans, tokens, vocab, *, max_distance=10):
    n_tokens = len(tokens)
    m_sequence = [0] * n_tokens
    m_locations = []
    for m_span in mention_spans:
        i = m_span[0]
        while i <= m_span[1]:
            m_sequence[i] = 1
            m_locations.append(i)
            i += 1

    m_locations = np.array(m_locations)
    rel_position_sequence = []
    for i in range(n_tokens):
        if len(m_locations) > 0:
            distance = m_locations - i
            arg_min_distance = np.argmin(np.abs(distance))
            rel_position = distance[arg_min_distance]
            if rel_position < -max_distance:
                val = str(-max_distance)
                rel_position_sequence.append(val)
                if vocab is not None:
                    vocab.add(val)
            elif max_distance < rel_position:
                val = str(max_distance)
                rel_position_sequence.append(val)
                if vocab is not None:
                    vocab.add(val)
            else:
                val = str(rel_position)
                rel_position_sequence.append(val)
                if vocab is not None:
                    vocab.add(val)
        else:
            if vocab is not None:
                vocab.add('NO_MENTION')
            rel_position_sequence.append('NO_MENTION')

    # This is for the EOS token
    rel_position_sequence.append('NO_MENTION')

    return rel_position_sequence


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


def clean_question(text):
    return re.sub(
        '\s+', ' ',
        re.sub(r'[~\*\(\)]|--', ' ', text)
    ).strip()


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
        x_train.append(nlp(clean_question(x)))
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
        x_test.append(nlp(clean_question(x)))

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

    w_indices.append(embedding_lookup.word[EOS])
    pos_indices.append(embedding_lookup.pos[EOS])
    iob_indices.append(embedding_lookup.iob[EOS])
    ent_type_indices.append(embedding_lookup.ent_type[EOS])

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
                UNK: 1,
                'EOS': 2
            }
            for i, term in enumerate(multi_vocab.pos, start=3):
                pos_lookup[term] = i

            iob_lookup = {
                'MASK': 0,
                UNK: 1,
                'EOS': 2,
            }
            for i, term in enumerate(multi_vocab.iob, start=3):
                iob_lookup[term] = i

            ent_type_lookup = {
                'MASK': 0,
                UNK: 1,
                'EOS': 2
            }
            for i, term in enumerate(multi_vocab.ent_type, start=3):
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
    x_batch_mention_padded = pad_batch(x_mention_batch, max_length)

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
    def __init__(self, batch_size, multi_embedding_lookup: MultiEmbeddingLookup,
                 rel_position_vocab, rel_position_lookup,
                 x_tokens, y_array, truncate=True, shuffle=True, train=True):
        self.train = train
        self.x_array_w = []
        self.x_array_pos = []
        self.x_array_iob = []
        self.x_array_ent_type = []
        self.x_array_mention = []
        self.y_array = []
        self.rel_position_vocab = rel_position_vocab
        self.rel_position_lookup = rel_position_lookup

        rel_position_tags = []
        for q in x_tokens:
            w_indicies, pos_indices, iob_indices, ent_type_indices = convert_tokens_to_representations(
                q, multi_embedding_lookup
            )
            this_mention_spans = extract_this_mentions(q)
            pronoun_mention_spans = [(i, i) for i in extract_pronoun_mentions(q)]
            title_mention_spans = extract_title_mentions(q)
            mention_spans = this_mention_spans + pronoun_mention_spans + title_mention_spans

            mention_tags = mentions_to_sequence(
                mention_spans, q, self.rel_position_vocab if train else None
            )
            rel_position_tags.append(mention_tags)

            self.x_array_w.append(w_indicies)
            self.x_array_pos.append(pos_indices)
            self.x_array_iob.append(iob_indices)
            self.x_array_ent_type.append(ent_type_indices)

        if train:
            for i, tag in enumerate(self.rel_position_vocab, start=2):
                self.rel_position_lookup[tag] = i

        for tag_list in rel_position_tags:
            mention_indices = []
            for t in tag_list:
                if t in self.rel_position_lookup:
                    mention_indices.append(self.rel_position_lookup[t])
                else:
                    mention_indices.append(self.rel_position_lookup[UNK])

            self.x_array_mention.append(mention_indices)

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
                self.x_array_mention[i].append(self.rel_position_lookup[UNK])

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
    def __init__(self):
        super(RnnEntityGuesser, self).__init__()
        guesser_conf = conf['guessers']['EntityRNN']
        self.features = set(guesser_conf['features'])

        self.max_epochs = guesser_conf['max_epochs']
        self.batch_size = guesser_conf['batch_size']
        self.learning_rate = guesser_conf['learning_rate']
        self.max_grad_norm = guesser_conf['max_grad_norm']
        self.rnn_type = guesser_conf['rnn_type']
        self.dropout_prob = guesser_conf['dropout_prob']
        self.bidirectional = guesser_conf['bidirectional']
        self.n_hidden_units = guesser_conf['n_hidden_units']
        self.n_hidden_layers = guesser_conf['n_hidden_layers']
        self.use_wiki = guesser_conf['use_wiki']
        self.use_triviaqa = guesser_conf['use_triviaqa']
        self.use_tagme = guesser_conf['use_tagme']
        self.sm_dropout_prob = guesser_conf['sm_dropout_prob']
        self.sm_dropout_before_linear = guesser_conf['sm_dropout_before_linear']
        self.n_tagme_sentences = guesser_conf['n_tagme_sentences']
        self.n_wiki_sentences = guesser_conf['n_wiki_sentences']
        self.use_cove = guesser_conf['use_cove']
        self.variational_dropout_prob = guesser_conf['variational_dropout_prob']
        self.use_locked_dropout = guesser_conf['use_locked_dropout']
        self.hyper_opt = guesser_conf['hyper_opt']
        self.hyper_opt_steps = guesser_conf['hyper_opt_steps']
        self.weight_decay = guesser_conf['weight_decay']

        self.class_to_i = None
        self.i_to_class = None
        self.vocab = None
        self.word_embeddings = None
        self.multi_embedding_lookup = None
        self.rel_position_vocab = None
        self.rel_position_lookup = None
        self.n_classes = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.nlp = None

    def parameters(self):
        return {
            'features': self.features,
            'max_epochs': self.max_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'max_grad_norm': self.max_grad_norm,
            'rnn_type': self.rnn_type,
            'dropout_prob': self.dropout_prob,
            'bidirectional': self.bidirectional,
            'n_hidden_units': self.n_hidden_units,
            'n_hidden_layers': self.n_hidden_layers,
            'use_wiki': self.use_wiki,
            'use_triviaqa': self.use_triviaqa,
            'use_tagme': self.use_tagme,
            'sm_dropout_prob': self.sm_dropout_prob,
            'sm_dropout_before_linear': self.sm_dropout_before_linear,
            'n_tagme_sentences': self.n_tagme_sentences,
            'n_wiki_sentences': self.n_wiki_sentences,
            'use_cove': self.use_cove,
            'variational_dropout_prob': self.variational_dropout_prob,
            'use_locked_dropout': self.use_locked_dropout,
            'weight_decay': self.weight_decay
        }

    def guess(self,
              questions: List[QuestionText],
              max_n_guesses: Optional[int]):
        x_test_tokens = [self.nlp(clean_question(x)) for x in questions]
        y_test = np.zeros(len(questions))
        dataset = BatchedDataset(
            self.batch_size, self.multi_embedding_lookup, self.rel_position_vocab, self.rel_position_lookup,
            x_test_tokens, y_test,
            truncate=False, shuffle=False, train=False
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
        self.nlp = spacy.load('en')
        x_train_tokens, y_train, x_test_tokens, y_test, vocab, class_to_i, i_to_class = preprocess_dataset(
            self.nlp, training_data
        )

        if self.use_wiki:
            wiki_dataset = WikipediaDataset(
                set(i_to_class), n_sentences=self.n_wiki_sentences, replace_title_mentions=WIKI_TITLE_MENTION
            )
            wiki_train_data = wiki_dataset.training_data()
            w_x_train_text, w_train_y, *_ = preprocess_dataset(
                self.nlp, wiki_train_data, train_size=1, vocab=vocab, class_to_i=class_to_i, i_to_class=i_to_class
            )
            log.info(f'Adding {len(w_x_train_text)} Wikipedia sentences as training data')
            x_train_tokens.extend(w_x_train_text)
            y_train.extend(w_train_y)

        if self.use_tagme:
            wiki_dataset = TagmeWikipediaDataset(n_examples=self.n_tagme_sentences)
            wiki_train_data = wiki_dataset.training_data()
            w_x_train_text, w_train_y, *_ = preprocess_dataset(
                self.nlp, wiki_train_data, train_size=1, vocab=vocab, class_to_i=class_to_i, i_to_class=i_to_class
            )
            log.info(f'Adding {len(w_x_train_text)} Tagme Wikipedia sentences as training data')
            x_train_tokens.extend(w_x_train_text)
            y_train.extend(w_train_y)

        if self.use_triviaqa:
            tqa_dataset = TriviaQADataset(set(i_to_class))
            tqa_train_data = tqa_dataset.training_data()
            tqa_x_train_text, tqa_train_y, *_ = preprocess_dataset(
                self.nlp, tqa_train_data, train_size=1, vocab=vocab, class_to_i=class_to_i, i_to_class=i_to_class
            )
            log.info(f'Adding {len(tqa_x_train_text)} TriviaQA examples as training data')
            x_train_tokens.extend(tqa_x_train_text)
            y_train.extend(tqa_train_y)

        self.class_to_i = class_to_i
        self.i_to_class = i_to_class
        self.vocab = vocab

        log.info('Loading word embeddings')
        word_embeddings, multi_embedding_lookup = load_multi_embeddings(multi_vocab=vocab)
        self.word_embeddings = word_embeddings
        self.multi_embedding_lookup = multi_embedding_lookup


        log.info('Batching the dataset')
        self.rel_position_vocab = set()
        self.rel_position_lookup = {
            'MASK': 0,
            UNK: 1
        }
        train_dataset = BatchedDataset(
            self.batch_size, multi_embedding_lookup, self.rel_position_vocab, self.rel_position_lookup,
            x_train_tokens, y_train, train=True
        )
        test_dataset = BatchedDataset(
            self.batch_size, multi_embedding_lookup, self.rel_position_vocab, self.rel_position_lookup,
            x_test_tokens, y_test, train=False
        )
        self.n_classes = compute_n_classes(training_data[1])

        if self.hyper_opt:
            self.hyperparameter_optimize(train_dataset, test_dataset)
        else:
            self._fit(train_dataset, test_dataset)

    def hyperparameter_optimize(self, train_dataset, test_dataset):
        from advisor_client.client import AdvisorClient

        client = AdvisorClient()
        study_id = os.environ.get('QB_STUDY_ID')
        if study_id is None:
            study_config = {
                'goal': 'MAXIMIZE',
                'maxTrials': self.hyper_opt_steps,
                'maxParallelTrials': 1,
                'randomInitTrials': 1,
                'params': [
                    {
                        'parameterName': 'sm_dropout',
                        'type': 'DOUBLE',
                        'minValue': 0,
                        'maxValue': 1
                    },
                    {
                        'parameterName': 'nn_dropout',
                        'type': 'DOUBLE',
                        'minValue': 0,
                        'maxValue': 1
                    },
                    {
                        'parameterName': 'variational_dropout',
                        'type': 'DOUBLE',
                        'minValue': 0,
                        'maxValue': 1
                    }
                ]
            }

            study = client.create_study('rnn', study_config)
        else:
            study_id = int(study_id)
            study = client.get_study_by_id(study_id)
        is_done = False
        while not is_done:
            trial = client.get_suggestions(study.id, 1)[0]
            trial_params = json.loads(trial.parameter_values)
            acc_score = self._fit(train_dataset, test_dataset, hyper_params=trial_params)
            client.complete_trial_with_one_metric(trial, acc_score)
            best_trial = client.get_best_trial(study.id)
            log.info(f'Best Trial: {best_trial}')
            is_done = client.is_study_done(study.id)

        raise ValueError('Hyperparameter optimization done, exiting')

    def _fit(self, train_dataset, test_dataset, hyper_params=None):
        model_params = {
            'sm_dropout': self.sm_dropout_prob,
            'nn_dropout': self.dropout_prob,
            'variational_dropout': self.variational_dropout_prob
        }
        if hyper_params is not None:
            for k, v in hyper_params.items():
                model_params[k] = v

        log.info('Initializing neural model')
        self.model = RnnEntityModel(
            len(self.multi_embedding_lookup.word),
            len(self.multi_embedding_lookup.pos),
            len(self.multi_embedding_lookup.iob),
            len(self.multi_embedding_lookup.ent_type),
            len(self.rel_position_lookup),
            self.n_classes,
            enabled_features=self.features,
            embeddings=self.word_embeddings,
            rnn_type=self.rnn_type, bidirectional=self.bidirectional,
            dropout_prob=model_params['nn_dropout'],
            variational_dropout_prob=model_params['variational_dropout'],
            sm_dropout_prob=model_params['sm_dropout'],
            n_hidden_units=self.n_hidden_units, n_hidden_layers=self.n_hidden_layers,
            sm_dropout_before_linear=self.sm_dropout_before_linear,
            use_cove=self.use_cove, use_locked_dropout=self.use_locked_dropout
        ).cuda()
        log.info(f'Parameters:\n{pformat(self.parameters())}')
        log.info(f'Model:\n{repr(self.model)}')
        log.info(f'Hyper params:\n{pformat(model_params)}')
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate, weight_decay=self.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, verbose=True, mode='max')

        tb_experiment = ' '.join(f'{param}={value}' for param, value in [
            ('model', 'rnn_entity'),
            ('features', '-'.join(sorted(self.features))),
            ('dropout', self.dropout_prob),
            ('sm_dropout', self.sm_dropout_prob),
            ('sm_drop_before_linear', self.sm_dropout_before_linear),
            ('lr', self.learning_rate),
            ('hu', self.n_hidden_units),
            ('n_layers', self.n_hidden_layers),
            ('rnn', self.rnn_type),
            ('bidirectional', self.bidirectional),
            ('use_wiki', self.use_wiki),
            ('use_triviaqa', self.use_triviaqa),
            ('use_tagme', self.use_tagme),
            ('n_wiki_sentences', self.n_wiki_sentences),
            ('n_tagme_sentences', self.n_tagme_sentences),
            ('use_cove', self.use_cove),
            ('variational_dropout_prob', self.variational_dropout_prob),
            ('use_locked_dropout', self.use_locked_dropout),
            ('weight_decay', self.weight_decay)
        ])

        manager = TrainingManager([
            BaseLogger(log_func=log.info), TerminateOnNaN(),
            EarlyStopping(monitor='test_acc', patience=10, verbose=1), MaxEpochStopping(self.max_epochs),
            ModelCheckpoint(create_save_model(self.model), '/tmp/rnn_entity.pt', monitor='test_acc'),
            Tensorboard(tb_experiment)
        ])

        log.info('Starting training...')
        best_acc = 0.0
        while True:
            self.model.train()
            train_acc, train_loss, train_time = self.run_epoch(train_dataset, evaluate=False)

            self.model.eval()
            test_acc, test_loss, test_time = self.run_epoch(test_dataset, evaluate=True)
            best_acc = max(best_acc, test_acc)

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
        return best_acc

    def run_epoch(self, batched_dataset: BatchedDataset, evaluate=False):
        if evaluate:
            batch_order = range(batched_dataset.n_batches)
        else:
            batch_order = np.random.permutation(batched_dataset.n_batches)

        batch_accuracies = []
        batch_losses = []
        epoch_start = time.time()
        hidden_init = self.model.init_hidden(self.batch_size)
        for batch in batch_order:
            t_x_w_batch = Variable(batched_dataset.t_x_w_batches[batch], volatile=evaluate)
            t_x_pos_batch = Variable(batched_dataset.t_x_pos_batches[batch], volatile=evaluate)
            t_x_iob_batch = Variable(batched_dataset.t_x_iob_batches[batch], volatile=evaluate)
            t_x_type_batch = Variable(batched_dataset.t_x_type_batches[batch], volatile=evaluate)
            t_x_mention_batch = Variable(batched_dataset.t_x_mention_batches[batch], volatile=evaluate)
            length_batch = batched_dataset.length_batches[batch]
            t_y_batch = Variable(batched_dataset.t_y_batches[batch], volatile=evaluate)

            self.model.zero_grad()
            out, hidden = self.model(
                t_x_w_batch, t_x_pos_batch, t_x_iob_batch, t_x_type_batch, t_x_mention_batch,
                length_batch, hidden_init
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
                'max_grad_norm': self.max_grad_norm,
                'features': self.features,
                'rel_position_lookup': self.rel_position_lookup,
                'rel_position_vocab': self.rel_position_vocab,
                'rnn_type': self.rnn_type,
                'dropout_prob': self.dropout_prob,
                'bidirectional': self.bidirectional,
                'n_hidden_units': self.n_hidden_units,
                'n_hidden_layers': self.n_hidden_layers,
                'use_wiki': self.use_wiki,
                'use_triviaqa': self.use_triviaqa,
                'use_tagme': self.use_tagme,
                'n_tagme_sentences': self.n_tagme_sentences,
                'n_wiki_sentences': self.n_wiki_sentences,
                'sm_dropout_prob': self.sm_dropout_prob,
                'sm_dropout_before_linear': self.sm_dropout_before_linear,
                'variational_dropout_prob': self.variational_dropout_prob,
                'use_cove': self.use_cove,
                'use_locked_dropout': self.use_locked_dropout,
                'hyper_opt': self.hyper_opt,
                'hyper_opt_steps': self.hyper_opt_steps,
                'weight_decay': self.weight_decay
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
        guesser.nlp = spacy.load('en')
        guesser.features = params['features']
        guesser.rel_position_vocab = params['rel_position_vocab']
        guesser.rel_position_lookup = params['rel_position_lookup']
        guesser.rnn_type = params['rnn_type']
        guesser.dropout_prob = params['dropout_prob']
        guesser.bidirectional = params['bidirectional']
        guesser.n_hidden_units = params['n_hidden_units']
        guesser.n_hidden_layers = params['n_hidden_layers']
        guesser.use_wiki = params['use_wiki']
        guesser.use_triviaqa = params['use_triviaqa']
        guesser.use_tagme = params['use_tagme']
        guesser.n_tagme_sentences = params['n_tagme_sentences']
        guesser.n_wiki_sentences = params['n_wiki_sentences']
        guesser.sm_dropout_prob = params['sm_dropout_prob']
        guesser.sm_dropout_before_linear = params['sm_dropout_before_linear']
        guesser.use_cove = params['use_cove']
        guesser.variational_dropout_prob = params['variational_dropout_prob']
        guesser.use_locked_dropout = params['use_locked_dropout']
        guesser.hyper_opt = params['hyper_opt']
        guesser.hyper_opt_steps = params['hyper_opt_steps']
        guesser.weight_decay = params['weight_decay']
        return guesser

    @classmethod
    def targets(cls):
        return ['rnn_entity.pickle', 'rnn_entity.pt']

    def qb_dataset(self):
        return QuizBowlDataset(guesser_train=True)


class RnnEntityModel(nn.Module):
    def __init__(self,
                 word_vocab_size, pos_vocab_size, iob_vocab_size, type_vocab_size, mention_vocab_size,
                 n_classes, embedding_dim=300, dropout_prob=.5, sm_dropout_prob=.3, sm_dropout_before_linear=True,
                 n_hidden_layers=1, n_hidden_units=1000, bidirectional=True, rnn_type='gru',
                 variational_dropout_prob=.5, enabled_features={'word', 'pos', 'iob', 'type', 'mention'},
                 embeddings=None, use_cove=False, use_locked_dropout=False):
        super(RnnEntityModel, self).__init__()
        self.word_vocab_size = word_vocab_size
        self.pos_vocab_size = pos_vocab_size
        self.iob_vocab_size = iob_vocab_size
        self.type_vocab_size = type_vocab_size
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout_prob
        self.sm_dropout_prob = sm_dropout_prob
        self.variational_dropout_prob = variational_dropout_prob
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_units = n_hidden_units
        self.bidirectional = bidirectional
        self.enabled_features = enabled_features
        self.sm_dropout_before_linear = sm_dropout_before_linear
        self.use_cove = use_cove
        self.use_locked_dropout = use_locked_dropout

        if use_locked_dropout:
            self.dropout = LockedDropout()
        else:
            self.dropout = nn.Dropout(dropout_prob)
        self.feature_dimension = 0

        if 'word' in enabled_features:
            self.word_embeddings = nn.Embedding(word_vocab_size, embedding_dim, padding_idx=0)
            self.feature_dimension += 300
        else:
            self.word_embeddings = None

        if 'pos' in enabled_features:
            self.pos_embeddings = nn.Embedding(pos_vocab_size, 50, padding_idx=0)
            self.feature_dimension += 50
        else:
            self.pos_embeddings = None

        if 'iob' in enabled_features:
            self.iob_embeddings = nn.Embedding(iob_vocab_size, 50, padding_idx=0)
            self.feature_dimension += 50
        else:
            self.iob_embeddings = None

        if 'type' in enabled_features:
            self.type_embeddings = nn.Embedding(type_vocab_size, 50, padding_idx=0)
            self.feature_dimension += 50
        else:
            self.type_embeddings = None

        if 'mention' in enabled_features:
            self.mention_embeddings = nn.Embedding(mention_vocab_size, 50, padding_idx=0)
            self.feature_dimension += 50
        else:
            self.mention_embeddings = None

        if embeddings is not None:
            self.word_embeddings.weight.data = torch.from_numpy(embeddings).float()

        if use_cove:
            from cove import MTLSTM
            self.cove = MTLSTM(n_vocab=embeddings.shape[0], vectors=torch.from_numpy(embeddings).float())
            self.cove.requires_grad = False
            for p in self.cove.parameters():
                p.requires_grad = False
            self.feature_dimension += 600
        else:
            self.cove = None

        self.rnn_type = rnn_type
        if rnn_type == 'lstm':
            rnn_layer = nn.LSTM
        elif rnn_type == 'gru':
            rnn_layer = nn.GRU
        else:
            raise ValueError('Unrecognized rnn layer type')

        rnn = rnn_layer(
            self.feature_dimension, n_hidden_units, n_hidden_layers,
            dropout=dropout_prob, batch_first=True, bidirectional=bidirectional
        )
        if variational_dropout_prob > 0:
            self.rnn = WeightDrop(rnn, ['weight_hh_l0'], dropout=variational_dropout_prob)
        else:
            self.rnn = rnn

        self.num_directions = int(bidirectional) + 1
        if sm_dropout_before_linear:
            self.classification_layer = nn.Sequential(
                nn.Dropout(sm_dropout_prob),
                nn.Linear(n_hidden_units * self.num_directions, n_classes),
                nn.BatchNorm1d(n_classes),
            )
        else:
            self.classification_layer = nn.Sequential(
                nn.Linear(n_hidden_units * self.num_directions, n_classes),
                nn.BatchNorm1d(n_classes),
                nn.Dropout(sm_dropout_prob)
            )

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.rnn_type == 'lstm':
            return (
                Variable(weight.new(self.n_hidden_layers * self.num_directions, batch_size, self.n_hidden_units).zero_()),
                Variable(weight.new(self.n_hidden_layers * self.num_directions, batch_size, self.n_hidden_units).zero_())
            )
        else:
            return Variable(weight.new(self.n_hidden_layers * self.num_directions, batch_size, self.n_hidden_units).zero_())

    def forward(self, word_idxs, pos_idxs, iob_idxs, type_idxs: Variable, mention_idxs, lengths, hidden):
        dropout_features = []
        if 'word' in self.enabled_features:
            word_embeddings = self.word_embeddings(word_idxs)
            dropout_features.append(word_embeddings)

        if 'pos' in self.enabled_features:
            pos_embeddings = self.pos_embeddings(pos_idxs)
            dropout_features.append(pos_embeddings)

        if 'iob' in self.enabled_features:
            iob_embeddings = self.iob_embeddings(iob_idxs)
            dropout_features.append(iob_embeddings)

        if 'type' in self.enabled_features:
            type_embeddings = self.type_embeddings(type_idxs)
            dropout_features.append(type_embeddings)

        if 'mention' in self.enabled_features:
            mention_embeddings = self.mention_embeddings(mention_idxs)
            dropout_features.append(mention_embeddings)

        if self.use_cove:
            cove_embeddings = self.cove(word_idxs, torch.from_numpy(lengths).cuda())
            dropout_features.append(cove_embeddings)
        features = torch.cat(dropout_features, 2)
        if self.use_locked_dropout:
            features = self.dropout(features, self.dropout_prob)
        else:
            features = self.dropout(features)

        packed_input = nn.utils.rnn.pack_padded_sequence(features, lengths, batch_first=True)

        output, hidden = self.rnn(packed_input, hidden)

        if type(hidden) == tuple:
            final_hidden = hidden[0]
        else:
            final_hidden = hidden

        batch_size = word_idxs.data.shape[0]
        final_hidden = final_hidden.view(
            self.n_hidden_layers, self.num_directions, batch_size, self.n_hidden_units
        )[-1].view(self.num_directions, batch_size, self.n_hidden_units)
        h_reshaped = final_hidden.transpose(0, 1).contiguous().view(
            word_idxs.data.shape[0], self.num_directions * self.n_hidden_units
        )

        return self.classification_layer(h_reshaped), hidden
