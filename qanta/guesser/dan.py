import re
import os
import shutil
import time
import cloudpickle
from typing import List, Optional, Dict

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import Adam, lr_scheduler

from torchtext.data.field import Field
from torchtext.data.iterator import Iterator

from qanta import qlogging
from qanta.util.io import shell, get_tmp_filename
from qanta.torch.dataset import QuizBowl
from qanta.config import conf
from qanta.guesser.abstract import AbstractGuesser
from qanta.datasets.abstract import QuestionText
from qanta.torch import (
    BaseLogger, TerminateOnNaN, EarlyStopping, ModelCheckpoint,
    MaxEpochStopping, TrainingManager
)


log = qlogging.get(__name__)


CUDA = torch.cuda.is_available()


def create_save_model(model):
    def save_model(path):
        torch.save(model.state_dict(), path)
    return save_model


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


class DanEncoder(nn.Module):
    def __init__(self, embedding_dim, n_hidden_layers, n_hidden_units, dropout_prob):
        super(DanEncoder, self).__init__()
        encoder_layers = []
        for i in range(n_hidden_layers):
            if i == 0:
                input_dim = embedding_dim
            else:
                input_dim = n_hidden_units

            encoder_layers.extend([
                nn.Linear(input_dim, n_hidden_units),
                nn.BatchNorm1d(n_hidden_units),
                nn.ELU(),
                nn.Dropout(dropout_prob),
            ])
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x_array):
        return self.encoder(x_array)


class DanModel(nn.Module):
    def __init__(self, n_classes, *,
                 text_field=None,
                 unigram_field=None, bigram_field=None, trigram_field=None,
                 init_embeddings=True, emb_dim=300,
                 n_hidden_units=1000, n_hidden_layers=1, nn_dropout=.265,
                 pooling='avg'):
        super(DanModel, self).__init__()
        self.emb_dim = emb_dim
        self.n_classes = n_classes
        self.n_hidden_units = n_hidden_units
        self.n_hidden_layers = n_hidden_layers
        self.nn_dropout = nn_dropout
        self.pooling = pooling

        self.dropout = nn.Dropout(nn_dropout)

        if (text_field is not None) and (
                unigram_field is not None or bigram_field is not None or trigram_field is not None):
            raise ValueError('Textfield being not None and any ngram field being not None is not allowed')

        if text_field is None and unigram_field is None and bigram_field is None and trigram_field is None:
            raise ValueError('Must have at least one text field')

        if text_field is None:
            self.text_vocab_size = None
            self.text_embeddings = None
            self.text_field = None
        else:
            text_vocab = text_field.vocab
            self.text_vocab_size = len(text_vocab)
            text_pad_idx = text_vocab.stoi[text_field.pad_token]
            self.text_embeddings = nn.Embedding(self.text_vocab_size, emb_dim, padding_idx=text_pad_idx)
            self.text_field = text_field
            if init_embeddings:
                mean_emb = text_vocab.vectors.mean(0)
                text_vocab.vectors[text_vocab.stoi[text_field.unk_token]] = mean_emb
                self.text_embeddings.weight.data = text_vocab.vectors.cuda()

        if unigram_field is None:
            self.unigram_vocab_size = None
            self.unigram_embeddings = None
            self.unigram_field = None
        else:
            unigram_vocab = unigram_field.vocab
            self.unigram_vocab_size = len(unigram_vocab)
            unigram_pad_idx = unigram_vocab.stoi[unigram_field.pad_token]
            self.unigram_embeddings = nn.Embedding(self.unigram_vocab_size, emb_dim, padding_idx=unigram_pad_idx)
            self.unigram_field = unigram_field
            if init_embeddings:
                mean_emb = unigram_vocab.vectors.mean(0)
                unigram_vocab.vectors[unigram_vocab.stoi[unigram_field.unk_token]] = mean_emb
                self.unigram_embeddings.weight.data = unigram_vocab.vectors.cuda()

        if bigram_field is None:
            self.bigram_vocab_size = None
            self.bigram_embeddings = None
            self.bigram_field = None
        else:
            bigram_vocab = bigram_field.vocab
            self.bigram_vocab_size = len(bigram_vocab)
            bigram_pad_idx = bigram_vocab.stoi[bigram_field.pad_token]
            self.bigram_embeddings = nn.Embedding(self.bigram_vocab_size, emb_dim, padding_idx=bigram_pad_idx)
            self.bigram_field = bigram_field

        if trigram_field is None:
            self.trigram_vocab_size = None
            self.trigram_embeddings = None
            self.trigram_field = None
        else:
            trigram_vocab = trigram_field.vocab
            self.trigram_vocab_size = len(trigram_vocab)
            trigram_pad_idx = trigram_vocab.stoi[trigram_field.pad_token]
            self.trigram_embeddings = nn.Embedding(self.trigram_vocab_size, emb_dim, padding_idx=trigram_pad_idx)
            self.trigram_field = trigram_field

        if text_field is not None:
            n_fields = 1
        else:
            n_fields = 0
            if unigram_field is not None:
                n_fields += 1
            if bigram_field is not None:
                n_fields += 1
            if trigram_field is not None:
                n_fields += 1
        self.encoder = DanEncoder(n_fields * emb_dim, self.n_hidden_layers, self.n_hidden_units, self.nn_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.n_hidden_units, n_classes),
            nn.BatchNorm1d(n_classes),
            nn.Dropout(self.nn_dropout)
        )

    def _pool(self, embed, lengths, batch_size):
        if self.pooling == 'avg':
            return embed.sum(1) / lengths.view(batch_size, -1)
        elif self.pooling == 'max':
            emb_max, _ = torch.max(embed, 1)
            return emb_max
        else:
            raise ValueError(f'Unsupported pooling type f{self.pooling}, only avg and max are supported')

    def forward(self, input_: Dict[str, Variable], lengths: Dict, qanta_ids):
        """
        :param input_: [batch_size, seq_len] of word indices
        :param lengths: Length of each example
        :param qanta_ids: QB qanta_id if a qb question, otherwise -1 for wikipedia, used to get domain as source/target
        :return:
        """
        for key in lengths:
            if not isinstance(lengths[key], Variable):
                lengths[key] = Variable(lengths[key].float(), volatile=not self.training)

        if self.text_field is not None:
            text_input = input_['text']
            embed = self.text_embeddings(text_input)
            embed = self._pool(embed, lengths['text'].float(), text_input.size()[0])
            embed = self.dropout(embed)
            encoded = self.encoder(embed)
            return self.classifier(encoded)
        else:
            embedding_list = []
            if self.unigram_field is not None:
                unigram_input = input_['unigram']
                embed = self.unigram_embeddings(unigram_input)
                embed = self._pool(embed, lengths['unigram'].float, unigram_input.size()[0])
                embed = self.dropout(embed)
                embedding_list.append(embed)

            if self.bigram_field is not None:
                bigram_input = input_['bigram']
                embed = self.bigram_embeddings(bigram_input)
                embed = self._pool(embed, lengths['bigram'].float, bigram_input.size()[0])
                embed = self.dropout(embed)
                embedding_list.append(embed)

            if self.trigram_field is not None:
                trigram_input = input_['trigram']
                embed = self.trigram_embeddings(trigram_input)
                embed = self._pool(embed, lengths['trigram'].float, trigram_input.size()[0])
                embed = self.dropout(embed)
                embedding_list.append(embed)

            concat_embed = torch.cat(embedding_list, dim=1)
            encoded = self.encoder(concat_embed)
            return self.classifier(encoded)


class DanGuesser(AbstractGuesser):
    def __init__(self, config_num):
        super(DanGuesser, self).__init__(config_num)
        if self.config_num is not None:
            guesser_conf = conf['guessers']['qanta.guesser.dan.DanGuesser'][self.config_num]
            self.gradient_clip = guesser_conf['gradient_clip']
            self.n_hidden_units = guesser_conf['n_hidden_units']
            self.n_hidden_layers = guesser_conf['n_hidden_layers']
            self.nn_dropout = guesser_conf['nn_dropout']
            self.batch_size = guesser_conf['batch_size']
            self.use_wiki = guesser_conf['use_wiki']
            self.n_wiki_sentences = guesser_conf['n_wiki_sentences']
            self.wiki_title_replace_token = guesser_conf['wiki_title_replace_token']
            self.lowercase = guesser_conf['lowercase']

            self.combined_ngrams = guesser_conf['combined_ngrams']
            self.unigrams = guesser_conf['unigrams']
            self.bigrams = guesser_conf['bigrams']
            self.trigrams = guesser_conf['trigrams']
            self.combined_max_vocab_size = guesser_conf['combined_max_vocab_size']
            self.unigram_max_vocab_size = guesser_conf['unigram_max_vocab_size']
            self.bigram_max_vocab_size = guesser_conf['bigram_max_vocab_size']
            self.trigram_max_vocab_size = guesser_conf['trigram_max_vocab_size']
            self.pooling = guesser_conf['pooling']

            self.random_seed = guesser_conf['random_seed']

        self.page_field: Optional[Field] = None
        self.qanta_id_field: Optional[Field] = None
        self.text_field: Optional[Field] = None
        self.unigram_field: Optional[Field] = None
        self.bigram_field: Optional[Field] = None
        self.trigram_field: Optional[Field] = None
        self.n_classes = None
        self.emb_dim = None
        self.model_file = None

        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None

    @property
    def ans_to_i(self):
        return self.page_field.vocab.stoi

    @property
    def i_to_ans(self):
        return self.page_field.vocab.itos

    def parameters(self):
        return conf['guessers']['qanta.guesser.dan.DanGuesser'][self.config_num]

    def train(self, training_data):
        log.info('Loading Quiz Bowl dataset')
        train_iter, val_iter, dev_iter = QuizBowl.iters(
            batch_size=self.batch_size, lower=self.lowercase,
            use_wiki=self.use_wiki, n_wiki_sentences=self.n_wiki_sentences,
            replace_title_mentions=self.wiki_title_replace_token,
            combined_ngrams=self.combined_ngrams, unigrams=self.unigrams, bigrams=self.bigrams, trigrams=self.trigrams,
            combined_max_vocab_size=self.combined_max_vocab_size,
            unigram_max_vocab_size=self.unigram_max_vocab_size,
            bigram_max_vocab_size=self.bigram_max_vocab_size,
            trigram_max_vocab_size=self.trigram_max_vocab_size
        )
        log.info(f'N Train={len(train_iter.dataset.examples)}')
        log.info(f'N Test={len(val_iter.dataset.examples)}')
        fields: Dict[str, Field] = train_iter.dataset.fields
        self.page_field = fields['page']
        self.n_classes = len(self.ans_to_i)
        self.qanta_id_field = fields['qanta_id']
        self.emb_dim = 300

        if 'text' in fields:
            self.text_field = fields['text']
            log.info(f'Text Vocab={len(self.text_field.vocab)}')
        if 'unigram' in fields:
            self.unigram_field = fields['unigram']
            log.info(f'Unigram Vocab={len(self.unigram_field.vocab)}')
        if 'bigram' in fields:
            self.bigram_field = fields['bigram']
            log.info(f'Bigram Vocab={len(self.bigram_field.vocab)}')
        if 'trigram' in fields:
            self.trigram_field = fields['trigram']
            log.info(f'Trigram Vocab={len(self.trigram_field.vocab)}')

        log.info('Initializing Model')
        self.model = DanModel(
            self.n_classes,
            text_field=self.text_field,
            unigram_field=self.unigram_field, bigram_field=self.bigram_field, trigram_field=self.trigram_field,
            emb_dim=self.emb_dim,
            n_hidden_units=self.n_hidden_units, n_hidden_layers=self.n_hidden_layers,
            nn_dropout=self.nn_dropout,
            pooling=self.pooling
        )
        if CUDA:
            self.model = self.model.cuda()
        log.info(f'Parameters:\n{self.parameters()}')
        log.info(f'Model:\n{self.model}')
        self.optimizer = Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, verbose=True, mode='max')

        temp_prefix = get_tmp_filename()
        self.model_file = f'{temp_prefix}.pt'
        manager = TrainingManager([
            BaseLogger(log_func=log.info), TerminateOnNaN(), EarlyStopping(monitor='test_acc', patience=10, verbose=1),
            MaxEpochStopping(100), ModelCheckpoint(create_save_model(self.model), self.model_file, monitor='test_acc')
        ])

        log.info('Starting training')

        epoch = 0
        while True:
            self.model.train()
            train_acc, train_loss, train_time = self.run_epoch(train_iter)

            self.model.eval()
            test_acc, test_loss, test_time = self.run_epoch(val_iter)

            stop_training, reasons = manager.instruct(
                train_time, train_loss, train_acc,
                test_time, test_loss, test_acc
            )

            if stop_training:
                log.info(' '.join(reasons))
                break
            else:
                self.scheduler.step(test_acc)
            epoch += 1

    def run_epoch(self, iterator: Iterator):
        is_train = iterator.train
        batch_accuracies = []
        batch_losses = []
        epoch_start = time.time()
        for batch in iterator:
            input_dict = {}
            lengths_dict = {}
            if hasattr(batch, 'text'):
                text, lengths = batch.text
                input_dict['text'] = text
                lengths_dict['text'] = lengths

            if hasattr(batch, 'unigram'):
                text, lengths = batch.unigram
                input_dict['unigram'] = text
                lengths_dict['unigram'] = lengths

            if hasattr(batch, 'bigram'):
                text, lengths = batch.bigram
                input_dict['bigram'] = text
                lengths_dict['bigram'] = lengths

            if hasattr(batch, 'trigram'):
                text, lengths = batch.trigram
                input_dict['trigram'] = text
                lengths_dict['trigram'] = lengths

            page = batch.page
            qanta_ids = batch.qanta_id.cuda()

            if is_train:
                self.model.zero_grad()

            out = self.model(input_dict, lengths_dict, qanta_ids)
            _, preds = torch.max(out, 1)
            accuracy = torch.mean(torch.eq(preds, page).float()).data[0]
            batch_loss = self.criterion(out, page)
            if is_train:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()

            batch_accuracies.append(accuracy)
            batch_losses.append(batch_loss.data[0])

        epoch_end = time.time()

        return np.mean(batch_accuracies), np.mean(batch_losses), epoch_end - epoch_start

    def guess(self, questions: List[QuestionText], max_n_guesses: Optional[int]):
        if len(questions) == 0:
            return []
        batch_size = 500
        if len(questions) < batch_size:
            return self._guess_batch(questions, max_n_guesses)
        else:
            all_guesses = []
            for i in range(0, len(questions), batch_size):
                batch_questions = questions[i:i + batch_size]
                guesses = self._guess_batch(batch_questions, max_n_guesses)
                all_guesses.extend(guesses)
            return all_guesses

    def _guess_batch(self, questions: List[QuestionText], max_n_guesses: Optional[int]):
        if len(questions) == 0:
            return []
        input_dict = {}
        lengths_dict = {}
        if self.text_field is not None:
            examples = [self.text_field.preprocess(q) for q in questions]
            text, lengths = self.text_field.process(examples, None, False)
            input_dict['text'] = text
            lengths_dict['text'] = lengths
        if self.unigram_field is not None:
            examples = [self.unigram_field.preprocess(q) for q in questions]
            text, lengths = self.unigram_field.process(examples, None, False)
            input_dict['unigram'] = text
            lengths_dict['unigram'] = lengths
        if self.bigram_field is not None:
            examples = [self.bigram_field.preprocess(q) for q in questions]
            text, lengths = self.bigram_field.process(examples, None, False)
            input_dict['bigram'] = text
            lengths_dict['bigram'] = lengths
        if self.trigram_field is not None:
            examples = [self.trigram_field.preprocess(q) for q in questions]
            text, lengths = self.trigram_field.process(examples, None, False)
            input_dict['trigram'] = text
            lengths_dict['trigram'] = lengths
        qanta_ids = self.qanta_id_field.process([0 for _ in questions]).cuda()
        guesses = []
        out = self.model(input_dict, lengths_dict, qanta_ids)
        probs = F.softmax(out).data.cpu().numpy()
        n_examples = probs.shape[0]
        preds = np.argsort(-probs, axis=1)
        for i in range(n_examples):
            guesses.append([])
            for p in preds[i][:max_n_guesses]:
                guesses[-1].append((self.i_to_ans[p], probs[i][p]))
        return guesses

    def save(self, directory: str):
        shutil.copyfile(self.model_file, os.path.join(directory, 'dan.pt'))
        shell(f'rm -f {self.model_file}')
        with open(os.path.join(directory, 'dan.pkl'), 'wb') as f:
            cloudpickle.dump({
                'page_field': self.page_field,
                'combined_text_field': self.text_field,
                'unigram_text_field': self.unigram_field,
                'bigram_text_field': self.bigram_field,
                'trigram_text_field': self.trigram_field,
                'combined_ngrams': self.combined_ngrams,
                'unigrams': self.unigrams,
                'bigrams': self.bigrams,
                'trigrams': self.trigrams,
                'combined_max_vocab_size': self.combined_max_vocab_size,
                'unigram_max_vocab_size': self.unigram_max_vocab_size,
                'bigram_max_vocab_size': self.bigram_max_vocab_size,
                'trigram_max_vocab_size': self.trigram_max_vocab_size,
                'qanta_id_field': self.qanta_id_field,
                'n_classes': self.n_classes,
                'gradient_clip': self.gradient_clip,
                'n_hidden_units': self.n_hidden_units,
                'n_hidden_layers': self.n_hidden_layers,
                'nn_dropout': self.nn_dropout,
                'batch_size': self.batch_size,
                'use_wiki': self.use_wiki,
                'n_wiki_sentences': self.n_wiki_sentences,
                'wiki_title_replace_token': self.wiki_title_replace_token,
                'lowercase': self.lowercase,
                'pooling': self.pooling,
                'random_seed': self.random_seed,
                'config_num': self.config_num
            }, f)

    @classmethod
    def load(cls, directory: str):
        with open(os.path.join(directory, 'dan.pkl'), 'rb') as f:
            params = cloudpickle.load(f)

        guesser = DanGuesser(params['config_num'])
        guesser.page_field = params['page_field']
        guesser.qanta_id_field = params['qanta_id_field']

        guesser.text_field = params['combined_text_field']
        guesser.unigram_field = params['unigram_text_field']
        guesser.bigram_field = params['bigram_text_field']
        guesser.trigram_field = params['trigram_text_field']

        guesser.combined_ngrams = params['combined_ngrams']
        guesser.unigrams = params['unigrams']
        guesser.bigrams = params['bigrams']
        guesser.trigrams = params['trigrams']

        guesser.combined_max_vocab_size = params['combined_max_vocab_size']
        guesser.unigram_max_vocab_size = params['unigram_max_vocab_size']
        guesser.bigram_max_vocab_size = params['bigram_max_vocab_size']
        guesser.trigram_max_vocab_size = params['trigram_max_vocab_size']

        guesser.n_classes = params['n_classes']
        guesser.gradient_clip = params['gradient_clip']
        guesser.n_hidden_units = params['n_hidden_units']
        guesser.n_hidden_layers = params['n_hidden_layers']
        guesser.nn_dropout = params['nn_dropout']
        guesser.use_wiki = params['use_wiki']
        guesser.n_wiki_sentences = params['n_wiki_sentences']
        guesser.wiki_title_replace_token = params['wiki_title_replace_token']
        guesser.lowercase = params['lowercase']
        guesser.pooling = params['pooling']
        guesser.random_seed = params['random_seed']
        guesser.model = DanModel(
            guesser.n_classes,
            text_field=guesser.text_field,
            unigram_field=guesser.unigram_field,
            bigram_field=guesser.bigram_field,
            trigram_field=guesser.trigram_field,
            init_embeddings=False, emb_dim=300,
            n_hidden_layers=guesser.n_hidden_layers,
            n_hidden_units=guesser.n_hidden_units,
            pooling=guesser.pooling
        )
        guesser.model.load_state_dict(torch.load(
            os.path.join(directory, 'dan.pt'), map_location=lambda storage, loc: storage
        ))
        guesser.model.eval()
        if CUDA:
            guesser.model = guesser.model.cuda()
        return guesser

    @classmethod
    def targets(cls):
        return ['dan.pt', 'dan.pkl']
