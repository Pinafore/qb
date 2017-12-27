import time
import pickle
import os
import shutil

import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn import functional as F
from torch.optim import Adam, lr_scheduler

from qanta import qlogging
from qanta.guesser.abstract import AbstractGuesser
from qanta.preprocess import preprocess_dataset, tokenize_question
from qanta.guesser.nn import create_load_embeddings_function, convert_text_to_embeddings_indices, compute_n_classes
from qanta.torch import (
    BaseLogger, TerminateOnNaN, Tensorboard,
    EarlyStopping, ModelCheckpoint, MaxEpochStopping, TrainingManager, create_save_model
)

log = qlogging.get(__name__)

PT_BCN_WE_TMP = '/tmp/qanta/deep/pt_bcn_we.pickle'
PT_BCN_WE = 'pt_bcn_we.pickle'
load_embeddings = create_load_embeddings_function(PT_BCN_WE_TMP, PT_BCN_WE, log)


def masked_softmax(vector, mask):
    result = F.softmax(vector)
    result = result * mask
    result = result / (result.sum(dim=1, keepdim=True) + 1e-13)
    return result


def create_mask(b_sents, b_lens):
    mask = np.zeros(b_sents.size(), dtype='float32')
    for z, curr_len in enumerate(b_lens):
        mask[z, :curr_len] = 1.
    mask = Variable(torch.from_numpy(mask).float().cuda())
    return mask


def to_torch_long(arr):
    return torch.from_numpy(arr).long().cuda()


def create_batch(x_array, y_array):
    lengths = np.array([len(r) for r in x_array])
    max_length = np.max(lengths)

    x_batch_padded = []
    for r in x_array:
        pad_r = list(r)
        while len(pad_r) < max_length:
            pad_r.append(0)
        x_batch_padded.append(pad_r)
    x_batch_padded = to_torch_long(np.array(x_batch_padded))
    lengths = to_torch_long(lengths)
    masks = create_mask(x_batch_padded, lengths)

    return (
        x_batch_padded,
        lengths,
        masks,
        to_torch_long(y_array)
    )


def batchify(batch_size, x_array, y_array, truncate=True, shuffle=True):
    n_examples = x_array.shape[0]
    n_batches = n_examples // batch_size
    if shuffle:
        random_order = np.random.permutation(n_examples)
        x_array = x_array[random_order]
        y_array = y_array[random_order]

    t_x_batches = []
    length_batches = []
    mask_batches = []
    t_y_batches = []

    for b in range(n_batches):
        x_batch = x_array[b * batch_size:(b + 1) * batch_size]
        y_batch = y_array[b * batch_size:(b + 1) * batch_size]
        x_batch, lengths, masks, y_batch = create_batch(x_batch, y_batch)

        t_x_batches.append(x_batch)
        length_batches.append(lengths)
        mask_batches.append(masks)
        t_y_batches.append(y_batch)

    if (not truncate) and (batch_size * n_batches < n_examples):
        x_batch = x_array[n_batches * batch_size:]
        y_batch = y_array[n_batches * batch_size:]

        x_batch, lengths, masks, y_batch = create_batch(x_batch, y_batch)

        t_x_batches.append(x_batch)
        length_batches.append(lengths)
        mask_batches.append(masks)
        t_y_batches.append(y_batch)

    return n_batches, t_x_batches, length_batches, mask_batches, t_y_batches


class BcnGuesser(AbstractGuesser):
    def __init__(self):
        super(BcnGuesser, self).__init__()
        self.batch_size = 256
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

    def train(self, training_data) -> None:
        log.info('Preprocessing data')
        x_train_text, y_train, x_test_text, y_test, vocab, class_to_i, i_to_class = preprocess_dataset(
            training_data
        )
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

        log.info('Batching data')
        n_batches_train, t_x_train, lengths_train, masks_train, t_y_train = batchify(
            self.batch_size, x_train, y_train, truncate=True
        )
        n_batches_test, t_x_test, lengths_test, masks_test, t_y_test = batchify(
            self.batch_size, x_test, y_test, truncate=False, shuffle=False
        )

        self.n_classes = compute_n_classes(training_data[1])

        log.info('Creating model')
        self.model = BCN(
            300, 500, embeddings.shape[0], self.n_classes,
            We=torch.from_numpy(embeddings)
        ).cuda()
        self.optimizer = Adam(self.model.parameters())
        self.criterion = nn.NLLLoss()

        log.info(f'Model:\n{self.model}')

        manager = TrainingManager([
            BaseLogger(log_func=log.info), TerminateOnNaN(),
            EarlyStopping(monitor='test_acc', patience=10, verbose=1), MaxEpochStopping(100),
            ModelCheckpoint(create_save_model(self.model), '/tmp/bcn.pt', monitor='test_acc'),
            Tensorboard('bcn')
        ])

        log.info('Starting training...')
        while True:
            self.model.train()
            train_acc, train_loss, train_time = self.run_epoch(
                n_batches_train,
                t_x_train, lengths_train, masks_train, t_y_train, evaluate=False
            )

            self.model.eval()
            test_acc, test_loss, test_time = self.run_epoch(
                n_batches_test,
                t_x_test, lengths_test, masks_test, t_y_test, evaluate=True
            )

            stop_training, reasons = manager.instruct(
                train_time, train_loss, train_acc,
                test_time, test_loss, test_acc
            )

            if stop_training:
                log.info(' '.join(reasons))
                break

    def run_epoch(self, n_batches, t_x_array, lengths_array, masks_array, t_y_array, evaluate=False):
        if evaluate:
            batch_order = range(n_batches)
        else:
            batch_order = np.random.permutation(n_batches)

        batch_accuracies = []
        batch_losses = []
        epoch_start = time.time()
        for batch in batch_order:
            t_x_batch = Variable(t_x_array[batch], volatile=evaluate)
            length_batch = lengths_array[batch]
            mask_batch = masks_array[batch]
            t_y_batch = Variable(t_y_array[batch], volatile=evaluate)

            self.model.zero_grad()
            out = self.model(t_x_batch, length_batch, mask_batch)
            _, preds = torch.max(out, 1)
            accuracy = torch.mean(torch.eq(preds, t_y_batch).float()).data[0]
            batch_loss = self.criterion(out, t_y_batch)
            if not evaluate:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), 5)
                self.optimizer.step()

            batch_accuracies.append(accuracy)
            batch_losses.append(batch_loss.data[0])

        epoch_end = time.time()

        return np.mean(batch_accuracies), np.mean(batch_losses), epoch_end - epoch_start


    def guess(self, questions, max_n_guesses):
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

        _, t_x_batches, lengths, masks, t_y_batches = batchify(
            self.batch_size, x_test, y_test,
            truncate=False, shuffle=False
        )

        self.model.eval()
        self.model.cuda()
        guesses = []
        for b in range(len(t_x_batches)):
            t_x = Variable(t_x_batches[b], volatile=True)
            length_batch = lengths[b]
            mask_batch = masks[b]

            probs = self.model(t_x, length_batch, mask_batch)
            scores, preds = torch.max(probs, 1)
            scores = scores.data.cpu().numpy()
            preds = preds.data.cpu().numpy()
            for p, s in zip(preds, scores):
                guesses.append([(self.i_to_class[p], s)])

        return guesses

    @classmethod
    def targets(cls):
        return ['bcn.pickle', 'bcn.pt']

    @classmethod
    def load(cls, directory: str):
        with open(os.path.join(directory, 'bcn.pickle'), 'rb') as f:
            params = pickle.load(f)

        guesser = BcnGuesser()
        guesser.vocab = params['vocab']
        guesser.class_to_i = params['class_to_i']
        guesser.i_to_class = params['i_to_class']
        guesser.embeddings = params['embeddings']
        guesser.embedding_lookup = params['embedding_lookup']
        guesser.n_classes = params['n_classes']
        guesser.batch_size = params['batch_size']
        guesser.model = torch.load(os.path.join(directory, 'bcn.pt'))
        return guesser

    def save(self, directory: str) -> None:
        shutil.copyfile('/tmp/bcn.pt', os.path.join(directory, 'bcn.pt'))
        with open(os.path.join(directory, 'bcn.pickle'), 'wb') as f:
            pickle.dump({
                'vocab': self.vocab,
                'class_to_i': self.class_to_i,
                'i_to_class': self.i_to_class,
                'embeddings': self.embeddings,
                'embedding_lookup': self.embedding_lookup,
                'n_classes': self.n_classes,
                'batch_size': self.batch_size
            }, f)


class BCN(nn.Module):
    def __init__(self, d_word, d_hid, len_voc,
        num_classes, We=None, freeze_embs=False):

        super(BCN, self).__init__()
        self.embedding = nn.Embedding(len_voc, d_word)

        # load pretrained
        if We is not None:
            self.embedding.weight.data.copy_(We)

        if freeze_embs:
            self.embedding.weight.requires_grad = False

        self.encoder = nn.LSTM(d_word, d_hid,
            bidirectional=True, num_layers=1, batch_first=True)

        self.encoder2 = nn.LSTM(d_hid * 2 * 3, d_hid,
            bidirectional=True, num_layers=1, batch_first=True)

        self.bd_dense_1 = nn.Linear(d_hid * 2 * 4, d_hid * 2 * 2)
        self.bd_dense_2 = nn.Linear(d_hid * 2 * 2, d_hid)
        self.bd_nonlin = F.relu

        self.out_dense = nn.Linear(d_hid, num_classes)
        self.out_nonlin = nn.LogSoftmax()
        self.drop = nn.Dropout(p=0.2)
        self.d_word = d_word
        self.d_hid = d_hid

        self.e_hid_init = Variable(torch.zeros(2, 1, self.d_hid).cuda())
        self.e_cell_init = Variable(torch.zeros(2, 1, self.d_hid).cuda())

        self.e_hid_init2 = Variable(torch.zeros(2, 1, self.d_hid).cuda())
        self.e_cell_init2 = Variable(torch.zeros(2, 1, self.d_hid).cuda())

        # attention params
        self.att_fn = masked_softmax

        self.V = nn.Parameter(torch.Tensor(self.d_hid * 2, 1).cuda())
        nn.init.xavier_uniform(self.V.data)


    def forward(self, sents, lens, mask):

        bsz, max_len = sents.size()
        embeds = self.embedding(sents)

        lens, indices = torch.sort(lens, 0, True)
        e_hid_init = self.e_hid_init.expand(2, bsz, self.d_hid).contiguous()
        e_cell_init = self.e_cell_init.expand(2, bsz, self.d_hid).contiguous()
        enc_states, _ = self.encoder(pack(embeds[indices], lens.tolist(), batch_first=True),
            (e_hid_init, e_cell_init))
        enc_states = unpack(enc_states, batch_first=True)[0]
        _, _indices = torch.sort(indices, 0)
        enc_states = enc_states[_indices]

        # compute self attention
        A = enc_states.bmm(enc_states.permute(0,2,1).contiguous())

        # flatten for softmax and then unflatten
        A = A.view(-1, max_len)
        A = self.att_fn(A, mask.unsqueeze(1).expand(bsz, max_len, max_len).contiguous().view(-1, max_len))
        A = A.view(bsz, max_len, max_len)

        # compute attn weighted average and feed to 2nd lstm
        C = A.bmm(enc_states)

        enc2_inp = torch.cat([enc_states, enc_states - C, enc_states * C], 2)
        e_hid_init2 = self.e_hid_init2.expand(2, bsz, self.d_hid).contiguous()
        e_cell_init2 = self.e_cell_init2.expand(2, bsz, self.d_hid).contiguous()
        enc_states2, _ = self.encoder2(enc2_inp, (e_hid_init2, e_cell_init2))

        # mask out hiddens
        enc_states2 = enc_states2 * mask[:, :, None]

        # do simple pools
        max_pool = torch.max(enc_states2, 1)[0]
        min_pool = torch.min(enc_states2, 1)[0]

        # normalize by lengths
        mean_pool = torch.sum(enc_states2, 1) / torch.sum(mask, 1)[:, None]

        # do self pooling
        self_weights = torch.mm(enc_states2.contiguous().view(-1, self.d_hid * 2), self.V).view(bsz, max_len)
        self_weights = self.att_fn(self_weights, mask)
        ave = self_weights[:, :, None] * enc_states2
        ave = torch.sum(ave, 1)

        # concat and pass thru relu feedforward net (no weird maxpool batchnormalized thing)
        concat = torch.cat([max_pool, min_pool, mean_pool, ave], 1)
        concat = self.drop(self.bd_nonlin(self.bd_dense_1(concat)))
        concat = self.drop(self.bd_nonlin(self.bd_dense_2(concat)))

        # softmax output layer
        preds = self.out_dense(concat)
        preds = self.drop(preds)
        preds = self.out_nonlin(preds)

        return preds