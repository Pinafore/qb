import re
import os
import shutil
import time
import pickle
import math
from typing import List, Optional, Dict

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import Adam, lr_scheduler, Optimizer

from torchtext.data.field import Field
from torchtext.data.iterator import Iterator

from qanta import qlogging
from qanta.torch.dataset import QuizBowl
from qanta.config import conf
from qanta.guesser.abstract import AbstractGuesser
from qanta.datasets.abstract import QuestionText
from qanta.torch import (
    BaseLogger, TerminateOnNaN, EarlyStopping, ModelCheckpoint,
    MaxEpochStopping, TrainingManager
)
from sklearn.neighbors import KDTree
import random

log = qlogging.get(__name__)


PT_RNN_WE_TMP = '/tmp/qanta/deep/pt_rnn_we.pickle'
PT_RNN_WE = 'pt_rnn_we.pickle'
CUDA = torch.cuda.is_available()


def create_save_model(model):
    def save_model(path):
        torch.save(model.state_dict(), path)
    return save_model

extracted_grads = {}
def extract_grad_hook(name):
    def hook(grad):
        extracted_grads[name] = grad
    return hook

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


class TiedModel(nn.Module):
    def __init__(self, text_field: Field, n_classes,
                 init_embeddings=True, emb_dim=300,
                 n_hidden_units=1000, n_hidden_layers=1, nn_dropout=.265, sm_dropout=.158):
        super(TiedModel, self).__init__()
        vocab = text_field.vocab
        self.vocab_size = len(vocab)
        self.emb_dim = emb_dim
        self.n_classes = n_classes
        self.n_hidden_units = n_hidden_units
        self.n_hidden_layers = n_hidden_layers
        self.nn_dropout = nn_dropout
        self.sm_dropout = sm_dropout

        self.dropout = nn.Dropout(nn_dropout)
        pad_idx = vocab.stoi[text_field.pad_token]
        self.general_embeddings = nn.Embedding(self.vocab_size, emb_dim, padding_idx=pad_idx)
        self.qb_embeddings = nn.Embedding(self.vocab_size, emb_dim, padding_idx=pad_idx)
        self.wiki_embeddings =nn.Embedding(self.vocab_size, emb_dim, padding_idx=pad_idx)
        qb_mask = torch.cat([torch.ones(1, 600), torch.zeros(1, 300)], dim=1)
        wiki_mask = torch.cat([torch.ones(1, 300), torch.zeros(1, 300), torch.ones(1, 300)], dim=1)
        self.combined_mask = torch.cat([qb_mask, wiki_mask], dim=0).float().cuda()

        if init_embeddings:
            mean_emb = vocab.vectors.mean(0)
            vocab.vectors[vocab.stoi[text_field.unk_token]] = mean_emb
            self.general_embeddings.weight.data = vocab.vectors.cuda()
            self.qb_embeddings.weight.data = vocab.vectors.cuda()
            self.wiki_embeddings.weight.data = vocab.vectors.cuda()

        # One averaged embedding for each of general, qb, and wiki
        self.encoder = DanEncoder(3 * emb_dim, self.n_hidden_layers, self.n_hidden_units, self.nn_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.n_hidden_units, n_classes),
            nn.BatchNorm1d(n_classes),
            nn.Dropout(self.sm_dropout)
        )

    def forward(self, input_: Variable, lengths, qnums):
        """
        :param input_: [batch_size, seq_len] of word indices
        :param lengths: Length of each example
        :param qnums: QB qnum if a qb question, otherwise -1 for wikipedia, used to get domain as source/target
        :return:
        """
        if not isinstance(lengths, Variable):
            lengths = Variable(lengths.float(), volatile=not self.training)

        g_embed = self.general_embeddings(input_)
        g_embed = g_embed.sum(1) / lengths.float().view(input_.size()[0], -1)
        g_embed = self.dropout(g_embed)

        qb_embed = self.qb_embeddings(input_)
        qb_embed = qb_embed.sum(1) / lengths.float().view(input_.size()[0], -1)
        qb_embed = self.dropout(qb_embed)

        wiki_embed = self.wiki_embeddings(input_)
        wiki_embed = wiki_embed.sum(1) / lengths.float().view(input_.size()[0], -1)
        wiki_embed = self.dropout(wiki_embed)

        # Need to use qnum to mask either qb or wiki embeddings here
        concat_embed = torch.cat([g_embed, qb_embed, wiki_embed], dim=1)
        mask = Variable(self.combined_mask[(qnums < 0).long()])
        masked_embed = concat_embed * mask

        encoded = self.encoder(masked_embed)
        return self.classifier(encoded)


class TiedGuesser(AbstractGuesser):
    def __init__(self):
        super(TiedGuesser, self).__init__()
        guesser_conf = conf['guessers']['Tied']
        self.gradient_clip = guesser_conf['gradient_clip']
        self.n_hidden_units = guesser_conf['n_hidden_units']
        self.n_hidden_layers = guesser_conf['n_hidden_layers']
        self.lr = guesser_conf['lr']
        self.nn_dropout = guesser_conf['nn_dropout']
        self.sm_dropout = guesser_conf['sm_dropout']
        self.batch_size = guesser_conf['batch_size']
        self.use_wiki = guesser_conf['use_wiki']
        self.n_wiki_sentences = guesser_conf['n_wiki_sentences']
        self.wiki_title_replace_token = guesser_conf['wiki_title_replace_token']
        self.lowercase = guesser_conf['lowercase']
        self.tied_l2 = guesser_conf['tied_l2']

        self.page_field: Optional[Field] = None
        self.qnum_field: Optional[Field] = None
        self.text_field: Optional[Field] = None
        self.n_classes = None
        self.emb_dim = None

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
        return conf['guessers']['Tied'].copy()

    def train(self, training_data):
        log.info('Loading Quiz Bowl dataset')
        train_iter, val_iter, dev_iter = QuizBowl.iters(
            batch_size=self.batch_size, lower=self.lowercase,
            use_wiki=self.use_wiki, n_wiki_sentences=self.n_wiki_sentences,
            replace_title_mentions=self.wiki_title_replace_token
        )
        log.info(f'N Train={len(train_iter.dataset.examples)}')
        log.info(f'N Test={len(val_iter.dataset.examples)}')
        fields: Dict[str, Field] = train_iter.dataset.fields
        self.page_field = fields['page']
        self.n_classes = len(self.ans_to_i)
        self.qnum_field = fields['qnum']
        self.text_field = fields['text']
        self.emb_dim = self.text_field.vocab.vectors.shape[1]
        log.info(f'Vocab={len(self.text_field.vocab)}')

        log.info('Initializing Model')
        self.model = TiedModel(
            self.text_field, self.n_classes, emb_dim=self.emb_dim,
            n_hidden_units=self.n_hidden_units, n_hidden_layers=self.n_hidden_layers,
            nn_dropout=self.nn_dropout, sm_dropout=self.sm_dropout,
        )
        if CUDA:
            self.model = self.model.cuda()
        log.info(f'Parameters:\n{self.parameters()}')
        log.info(f'Model:\n{self.model}')
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, verbose=True, mode='max')

        manager = TrainingManager([
            BaseLogger(log_func=log.info), TerminateOnNaN(), EarlyStopping(monitor='test_acc', patience=10, verbose=1),
            MaxEpochStopping(100), ModelCheckpoint(create_save_model(self.model), '/tmp/rnn.pt', monitor='test_acc')
        ])

        log.info('Starting training')
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

    def run_epoch(self, iterator: Iterator):
        is_train = iterator.train
        batch_accuracies = []
        batch_losses = []
        epoch_start = time.time()
        for batch in iterator:
            text, lengths = batch.text
            page = batch.page
            qnums = batch.qnum.cuda()

            if is_train:
                self.model.zero_grad()

            out = self.model(text, lengths, qnums)
            _, preds = torch.max(out, 1)
            accuracy = torch.mean(torch.eq(preds, page).float()).data[0]
            batch_loss = self.criterion(out, page)
            if self.tied_l2 != 0:
                w_general = self.model.general_embeddings.weight
                w_source = self.model.wiki_embeddings.weight
                w_target = self.model.qb_embeddings.weight
                tied_weight_l2 = self.tied_l2 / 2 * (
                        (w_general ** 2).sum() +
                        ((w_source - w_general) ** 2).sum() +
                        ((w_target - w_general) ** 2).sum()
                )
                batch_loss += tied_weight_l2
            if is_train:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()

            batch_accuracies.append(accuracy)
            batch_losses.append(batch_loss.data[0])

        epoch_end = time.time()

        return np.mean(batch_accuracies), np.mean(batch_losses), epoch_end - epoch_start

    def guess(self, questions: List[QuestionText], max_n_guesses: Optional[int]):
        examples = [self.text_field.preprocess(q) for q in questions]
        text, lengths = self.text_field.process(examples, None, False)
        qnums = self.qnum_field.process([0 for _ in questions]).cuda()
        guesses = []
        out = self.model(text, lengths, qnums)
        probs = F.softmax(out)
        scores, preds = torch.max(probs, 1)
        scores = scores.data.cpu().numpy()
        preds = preds.data.cpu().numpy()

        for p, s in zip(preds, scores):
            guesses.append([(self.i_to_ans[p], s)])

        return guesses


    def save(self, directory: str):
        shutil.copyfile('/tmp/rnn.pt', os.path.join(directory, 'rnn.pt'))
        with open(os.path.join(directory, 'rnn.pkl'), 'wb') as f:
            pickle.dump({
                'page_field': self.page_field,
                'text_field': self.text_field,
                'qnum_field': self.qnum_field,
                'n_classes': self.n_classes,
                'emb_dim': self.emb_dim,
                'gradient_clip': self.gradient_clip,
                'n_hidden_units': self.n_hidden_units,
                'n_hidden_layers': self.n_hidden_layers,
                'lr': self.lr,
                'nn_dropout': self.nn_dropout,
                'sm_dropout': self.sm_dropout,
                'batch_size': self.batch_size,
                'use_wiki': self.use_wiki,
                'n_wiki_sentences': self.n_wiki_sentences,
                'wiki_title_replace_token': self.wiki_title_replace_token,
                'lowercase': self.lowercase,
                'tied_l2': self.tied_l2
            }, f)

    @classmethod
    def load(cls, directory: str):
        with open(os.path.join(directory, 'rnn.pkl'), 'rb') as f:
            params = pickle.load(f)

        guesser = TiedGuesser()
        guesser.page_field = params['page_field']
        guesser.text_field = params['text_field']
        guesser.qnum_field = params['qnum_field']
        guesser.n_classes = params['n_classes']
        guesser.emb_dim = params['emb_dim']
        guesser.gradient_clip = params['gradient_clip']
        guesser.n_hidden_units = params['n_hidden_units']
        guesser.n_hidden_layers = params['n_hidden_layers']
        guesser.lr = params['lr']
        guesser.nn_dropout = params['nn_dropout']
        guesser.sm_dropout = params['sm_dropout']
        guesser.use_wiki = params['use_wiki']
        guesser.n_wiki_sentences = params['n_wiki_sentences']
        guesser.wiki_title_replace_token = params['wiki_title_replace_token']
        guesser.lowercase = params['lowercase']
        guesser.tied_l2 = params['tied_l2']
        guesser.model = TiedModel(
            guesser.text_field, guesser.n_classes,
            init_embeddings=False, emb_dim=guesser.emb_dim
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

    # Runs query through the model and computes gradient based attacks
    
    def attack(self, query):
        text = TEXT.preprocess(query)
        text, lengths = self.text_field.process([text], None, False)
        
        text = TEXT.preprocess(query)
        text = [[TEXT.vocab.stoi[x] for x in text]]
        x = TEXT.tensor_type(text)
        lengths = torch.FloatTensor([x.size()[1]]).cuda()
        x = Variable(x).cuda()
    
        qnums = self.qnum_field.process([0]).cuda()
        y = self.model(x, lengths, qnums, extract_grad_hook('g_embed'))
        label = torch.max(y, 1)[1] # assume prediction is correct
    
        #print(self.i_to_ans[label.data.cpu().numpy()[0]]) #make sure label is the same as when you guess
    
        loss = criterion(y, label)
        self.model.zero_grad()
        loss.backward()
    
        grads = extracted_grads['g_embed'].transpose(0, 1)
        grads = grads.data.cpu()            
        scores = grads.sum(dim=2).numpy()
        grads = grads.numpy()
        text = x.transpose(0, 1).data.cpu().numpy()
        y = y.data.cpu().numpy()
            
        scores = scores.tolist()
        sorted_scores = list(scores) # make copy
        sorted_scores.sort(reverse=True)
    
        # we want the line above for the general case, but for DAN all the gradients are the same for each word, so just pick some
        #order = [scores.index(index) for index in sorted_scores]        
        order = random.sample(range(x.size()[1]), n_replace)   
    
        returnVal = ""
        for j in order[:n_replace]:
            returnVal = returnVal + TEXT.vocab.itos[text[j][0]] + "*"
            old_embed = TEXT.vocab.vectors[text[j][0]].numpy()
    
            _, inds = tree.query([old_embed], k=2)
            repl = inds[0][0] if inds[0][0] != text[j] else inds[0][1]        
    
            returnVal = returnVal + TEXT.vocab.itos[repl.item()] + "**"
    
        return returnVal
    #return "Server*is**Not*Running**Currently*Please**Come*Back**Tomorrow*Thanks"

# Hyperparameters
n_replace = 5
eps = 10
norm = np.inf

# Load model
save_path = './output/guesser/qanta.guesser.tied.TiedGuesser/'
guesser = TiedGuesser.load(save_path)
TEXT = guesser.text_field

def attack(query):
    return guesser.attack(query)

# Create KD tree for nearest neighbor
tree = KDTree(TEXT.vocab.vectors.numpy())
print('KDTree built for {} words'.format(len(TEXT.vocab)))

criterion = nn.CrossEntropyLoss()

def to_numpy(x):
    if isinstance(x, Variable):
        x = x.data
    try:
        if x.is_cuda:
            x = x.cpu()
    except AttributeError:
        pass
    if isinstance(x, torch.LongTensor) or isinstance(x, torch.FloatTensor):
        x = x.numpy()
    return x

def to_sentence(words):
    words = to_numpy(words)
    words = [TEXT.vocab.itos[w] for w in words]
    words = [w for w in words if w != '<pad>']
    return ' '.join(words)

