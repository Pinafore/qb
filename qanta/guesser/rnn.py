import re
import math
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
from qanta.torch.dataset import QuizBowl, create_qb_tokenizer
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


class RnnModel(nn.Module):
    def __init__(self, n_classes, *,
                 text_field=None,
                 init_embeddings=True, emb_dim=300,
                 n_hidden_units=1000, n_hidden_layers=1,
                 nn_dropout=.265, bidirectional=True):
        super(RnnModel, self).__init__()
        self.emb_dim = emb_dim
        self.n_classes = n_classes
        self.n_hidden_units = n_hidden_units
        self.n_hidden_layers = n_hidden_layers
        self.nn_dropout = nn_dropout
        self.bidirectional = bidirectional
        self.num_directions = 1 + int(bidirectional)

        self.dropout = nn.Dropout(nn_dropout)

        text_vocab = text_field.vocab
        self.text_vocab_size = len(text_vocab)
        text_pad_idx = text_vocab.stoi[text_field.pad_token]
        self.text_embeddings = nn.Embedding(self.text_vocab_size, emb_dim, padding_idx=text_pad_idx)
        self.text_field = text_field
        if init_embeddings:
            mean_emb = text_vocab.vectors.mean(0)
            text_vocab.vectors[text_vocab.stoi[text_field.unk_token]] = mean_emb
            self.text_embeddings.weight.data = text_vocab.vectors.cuda()

        self.rnn = nn.GRU(
            self.emb_dim, n_hidden_units, n_hidden_layers,
            dropout=self.nn_dropout, batch_first=True, bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.num_directions * self.n_hidden_units, n_classes),
            nn.BatchNorm1d(n_classes),
            nn.Dropout(self.nn_dropout)
        )

    def forward(self,
                text_input: Variable,
                lengths: List[int],
                hidden: Variable,
                qanta_ids):
        """
        :param text_input: [batch_size, seq_len] of word indices
        :param lengths: Length of each example
        :param qanta_ids: QB qanta_id if a qb question, otherwise -1 for wikipedia, used to get domain as source/target
        :param hidden: hidden state
        :return:
        """
        embed = self.text_embeddings(text_input)
        embed = self.dropout(embed)

        packed_input = nn.utils.rnn.pack_padded_sequence(embed, lengths, batch_first=True)
        output, hidden = self.rnn(packed_input, hidden)

        if type(hidden) == tuple:
            final_hidden = hidden[0]
        else:
            final_hidden = hidden

        batch_size = text_input.data.shape[0]

        # Since number of layers is variable, we need a way to reduce this
        # to just one output. The easiest is to take the last hidden, but
        # we could try other things too.
        final_hidden = final_hidden.view(
            self.n_hidden_layers, self.num_directions, batch_size, self.n_hidden_units
        )[-1]
        combined_hidden = torch.cat([final_hidden[i] for i in range(self.num_directions)], dim=1)

        return self.classifier(combined_hidden), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if isinstance(self.rnn, nn.LSTM):
            return (
                Variable(weight.new(
                    self.n_hidden_layers * self.num_directions, batch_size, self.n_hidden_units).zero_()),
                Variable(
                    weight.new(self.n_hidden_layers * self.num_directions, batch_size, self.n_hidden_units).zero_())
            )
        else:
            return Variable(weight.new(
                self.n_hidden_layers * self.num_directions, batch_size, self.n_hidden_units).zero_())


class RnnGuesser(AbstractGuesser):
    def __init__(self, config_num):
        super(RnnGuesser, self).__init__(config_num)
        if self.config_num is not None:
            guesser_conf = conf['guessers']['qanta.guesser.rnn.RnnGuesser'][self.config_num]
            self.gradient_clip = guesser_conf['gradient_clip']
            self.n_hidden_units = guesser_conf['n_hidden_units']
            self.n_hidden_layers = guesser_conf['n_hidden_layers']
            self.nn_dropout = guesser_conf['dropout']
            self.batch_size = guesser_conf['batch_size']
            self.use_wiki = guesser_conf['use_wiki']
            self.n_wiki_sentences = guesser_conf['n_wiki_sentences']
            self.wiki_title_replace_token = guesser_conf['wiki_title_replace_token']
            self.lowercase = guesser_conf['lowercase']

            self.random_seed = guesser_conf['random_seed']

        self.page_field: Optional[Field] = None
        self.qanta_id_field: Optional[Field] = None
        self.text_field: Optional[Field] = None
        self.n_classes = None
        self.emb_dim = None
        self.model_file = None

        self.model: Optional[RnnModel] = None
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
        return conf['guessers']['qanta.guesser.rnn.RnnGuesser'][self.config_num]

    def train(self, training_data):
        log.info('Loading Quiz Bowl dataset')
        train_iter, val_iter, dev_iter = QuizBowl.iters(
            batch_size=self.batch_size, lower=self.lowercase,
            use_wiki=self.use_wiki, n_wiki_sentences=self.n_wiki_sentences,
            replace_title_mentions=self.wiki_title_replace_token,
            sort_within_batch=True
        )
        log.info(f'Training Data={len(training_data[0])}')
        log.info(f'N Train={len(train_iter.dataset.examples)}')
        log.info(f'N Test={len(val_iter.dataset.examples)}')
        fields: Dict[str, Field] = train_iter.dataset.fields
        self.page_field = fields['page']
        self.n_classes = len(self.ans_to_i)
        self.qanta_id_field = fields['qanta_id']
        self.emb_dim = 300

        self.text_field = fields['text']
        log.info(f'Text Vocab={len(self.text_field.vocab)}')

        log.info('Initializing Model')
        self.model = RnnModel(
            self.n_classes,
            text_field=self.text_field,
            emb_dim=self.emb_dim,
            n_hidden_units=self.n_hidden_units, n_hidden_layers=self.n_hidden_layers,
            nn_dropout=self.nn_dropout
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
        batch_size = self.batch_size
        hidden_init = self.model.init_hidden(batch_size)
        epoch_start = time.time()
        for batch in iterator:
            text, lengths = batch.text
            lengths = list(lengths.cpu().numpy())
            if len(lengths) != batch_size:
                batch_size = len(lengths)
                hidden_init = self.model.init_hidden(batch_size)

            page = batch.page
            qanta_ids = batch.qanta_id.cuda()

            if is_train:
                self.model.zero_grad()

            out, hidden = self.model(
                text, lengths, hidden_init, qanta_ids
            )
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
        batch_size = 128
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
        examples = [self.text_field.preprocess(q) for q in questions]
        padded_examples, lengths = self.text_field.pad(examples)
        padded_examples = np.array(padded_examples, dtype=np.object)
        lengths = np.array(lengths)
        order = np.argsort(-lengths)
        rev_order = np.argsort(order)
        ordered_examples = padded_examples[order]
        ordered_lengths = lengths[order]
        text, lengths = self.text_field.numericalize((ordered_examples, ordered_lengths), device=None, train=False)
        lengths = list(lengths.cpu().numpy())

        qanta_ids = self.qanta_id_field.process([0 for _ in questions]).cuda()
        guesses = []
        hidden_init = self.model.init_hidden(len(questions))
        out, _ = self.model(text, lengths, hidden_init, qanta_ids)
        ordered_probs = F.softmax(out).data.cpu().numpy()
        probs = ordered_probs[rev_order]

        n_examples = probs.shape[0]
        preds = np.argsort(-probs, axis=1)
        for i in range(n_examples):
            guesses.append([])
            for p in preds[i][:max_n_guesses]:
                guesses[-1].append((self.i_to_ans[p], probs[i][p]))
        return guesses

    def save(self, directory: str):
        shutil.copyfile(self.model_file, os.path.join(directory, 'rnn.pt'))
        shell(f'rm -f {self.model_file}')
        with open(os.path.join(directory, 'rnn.pkl'), 'wb') as f:
            cloudpickle.dump({
                'page_field': self.page_field,
                'text_field': self.text_field,
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
                'random_seed': self.random_seed,
                'config_num': self.config_num
            }, f)

    @classmethod
    def load(cls, directory: str):
        with open(os.path.join(directory, 'rnn.pkl'), 'rb') as f:
            params = cloudpickle.load(f)

        guesser = RnnGuesser(params['config_num'])
        guesser.page_field = params['page_field']
        guesser.qanta_id_field = params['qanta_id_field']

        guesser.text_field = params['text_field']

        guesser.n_classes = params['n_classes']
        guesser.gradient_clip = params['gradient_clip']
        guesser.n_hidden_units = params['n_hidden_units']
        guesser.n_hidden_layers = params['n_hidden_layers']
        guesser.nn_dropout = params['nn_dropout']
        guesser.use_wiki = params['use_wiki']
        guesser.n_wiki_sentences = params['n_wiki_sentences']
        guesser.wiki_title_replace_token = params['wiki_title_replace_token']
        guesser.lowercase = params['lowercase']
        guesser.random_seed = params['random_seed']
        guesser.model = RnnModel(
            guesser.n_classes,
            text_field=guesser.text_field,
            init_embeddings=False, emb_dim=300,
            n_hidden_layers=guesser.n_hidden_layers,
            n_hidden_units=guesser.n_hidden_units
        )
        guesser.model.load_state_dict(torch.load(
            os.path.join(directory, 'rnn.pt'), map_location=lambda storage, loc: storage
        ))
        guesser.model.eval()
        if CUDA:
            guesser.model = guesser.model.cuda()
        return guesser

    @classmethod
    def targets(cls):
        return ['rnn.pt', 'rnn.pkl']

    def web_api(self, host='0.0.0.0', port=6000, debug=False):
        from flask import Flask, jsonify, request
        app = Flask(__name__)

        @app.route('/api/answer_question', methods=['POST'])
        def answer_question_base():
            text = request.form['text']
            guess, score = self.guess([text], 1)[0][0]
            return jsonify({'guess': guess, 'score': float(score)})

        @app.route('/api/interface_get_highlights', methods=['POST'])
        def get_highlights():
            questions = [request.form['text']]
            examples = [self.text_field.preprocess(q) for q in questions]
            padded_examples, lengths = self.text_field.pad(examples)
            padded_examples = np.array(padded_examples, dtype=np.object)
            lengths = np.array(lengths)
            order = np.argsort(-lengths)
            # rev_order = np.argsort(order)
            ordered_examples = padded_examples[order]
            ordered_lengths = lengths[order]
            text, lengths = self.text_field.numericalize((ordered_examples, ordered_lengths), device=-1, train=False)
            lengths = list(lengths.cpu().numpy())

            qanta_ids = self.qanta_id_field.process([0 for _ in questions])  # .cuda()
            hidden_init = self.model.init_hidden(len(questions))
            text = Variable(text.data, volatile=False)

            out, _ = self.model(text, lengths, hidden_init, qanta_ids, extract_grad_hook('embed'))

            guessForEvidence = request.form['guessForEvidence']
            guessForEvidence = guessForEvidence.split("style=\"color:blue\">")[1].split("</a>")[0].lower()
            indicator = -1

            guess = str(guessForEvidence)
            guesses = self.guess([request.form['text']], 500)[0]
            for index, (g, s) in enumerate(guesses):
                print(g.lower().replace("_", " ")[0:25])
                print(guessForEvidence)
                if g.lower().replace("_", " ")[0:25] == guessForEvidence:
                    print("INDICATOR SET")
                    indicator = index
                    guess = g.lower().replace("_", " ")[0:25]
                    break
            if indicator == -1:
                highlights = {
                    'wiki': ['No Evidence', 'No Evidence'],
                    'qb': ['No Evidence', 'No Evidence'],
                    'guess': guess,
                    'visual': 'No Evidence'
                }
                return jsonify(highlights)

            # label = torch.max(out,1)[1]
            label = torch.topk(out, k=500, dim=1)
            label = label[1][0][indicator]  # [0]

            criterion = nn.CrossEntropyLoss()
            loss = criterion(out, label)
            self.model.zero_grad()
            loss.backward()

            grads = extracted_grads['embed'].transpose(0, 1)
            grads = grads.data.cpu()
            scores = grads.sum(dim=2).numpy()
            grads = grads.numpy()
            text = text.transpose(0, 1).data.cpu().numpy()

            scores = scores.tolist()

            normalized_scores = scores
            # normalize scores across the words, doing positive and negatives seperately        
            # final scores should be in range [0,1] 0 is dark red, 1 is dark blue. 0.5 is no highlight
            total_score_pos = 1e-6    # 1e-6 for case where all positive/neg scores are 0
            total_score_neg = 1e-6
            for idx, s in enumerate(normalized_scores):
                s[0] = s[0] * s[0] * s[0] / 5
                if s[0] < 0:
                    total_score_neg = total_score_neg + math.fabs(s[0])
                else:
                    total_score_pos = total_score_pos + s[0]
            for idx, s in enumerate(normalized_scores):
                if s[0] < 0:
                    normalized_scores[idx] = (s[0] / total_score_neg) / 2   # / by 2 to get max of -0.5
                else:
                    normalized_scores[idx] = 0.0
            normalized_scores = [0.5 + n for n in normalized_scores]  # center scores

            returnVal = ""
            for s in normalized_scores:
                returnVal = returnVal + ' ' + str(s)

            localPreprocess = create_qb_tokenizer()
            examples = [localPreprocess(q) for q in questions]
            words = []
            for t in examples[0]:
                words.append(str(t))

            visual = colorize(words, normalized_scores, colors='RdBu')
            print("Guess", guess)
            highlights = {
                'wiki': [returnVal, returnVal],
                'qb': [returnVal, returnVal],
                'guess': guess,
                'visual': visual
            }
            return jsonify(highlights)

        @app.route('/api/interface_answer_question', methods=['POST'])
        def answer_question():
            text = request.form['text']
            answer = request.form['answer']
            answer = answer.replace(" ", "_").lower()
            guesses = self.guess([text], 20)[0]
            score_fn = []
            sum_normalize = 0.0
            for (g, s) in guesses:
                exp = np.exp(3*float(s))
                score_fn.append(exp)
                sum_normalize += exp
            for index, (g, s) in enumerate(guesses):
                guesses[index] = (g, score_fn[index] / sum_normalize)

            guess = []
            score = []
            answer_found = False
            num = 0
            for index, (g, s) in enumerate(guesses):
                if index >= 5:
                    break
                guess.append(g)
                score.append(float(s))
            for gue in guess:
                if (gue.lower() == answer.lower()):
                    answer_found = True
                    num = -1
            if (not answer_found):
                for index, (g, s) in enumerate(guesses):
                    if (g.lower() == answer.lower()):
                        guess.append(g)
                        score.append(float(s))
                        num = index + 1
            if (num == 0):
                print("num was 0")
                if (request.form['bell'] == 'true'):
                    return "Num0"
            guess = [g.replace("_", " ") for g in guess]
            return jsonify({'guess': guess, 'score': score, 'num': num})
        app.run(host=host, port=port, debug=debug)
