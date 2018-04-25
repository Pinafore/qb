import os
import numpy as np
from tqdm import tqdm

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter

embed_init = chainer.initializers.Uniform(.25)


def sequence_embed(embed, xs, dropout=0.):
    """Efficient embedding function for variable-length sequences

    This output is equally to
    "return [F.dropout(embed(x), ratio=dropout) for x in xs]".
    However, calling the functions is one-shot and faster.

    Args:
        embed (callable): A :func:`~chainer.functions.embed_id` function
            or :class:`~chainer.links.EmbedID` link.
        xs (list of :class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): i-th element in the list is an input variable,
            which is a :math:`(L_i, )`-shaped int array.
        dropout (float): Dropout ratio.

    Returns:
        list of ~chainer.Variable: Output variables. i-th element in the
        list is an output variable, which is a :math:`(L_i, N)`-shaped
        float array. :math:`(N)` is the number of dimensions of word embedding.

    """
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    ex = F.dropout(ex, ratio=dropout)
    exs = F.split_axis(ex, x_section, 0)
    return exs


def block_embed(embed, x, dropout=0.):
    """Embedding function followed by convolution

    Args:
        embed (callable): A :func:`~chainer.functions.embed_id` function
            or :class:`~chainer.links.EmbedID` link.
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable, which
            is a :math:`(B, L)`-shaped int array. Its first dimension
            :math:`(B)` is assumed to be the *minibatch dimension*.
            The second dimension :math:`(L)` is the length of padded
            sentences.
        dropout (float): Dropout ratio.

    Returns:
        ~chainer.Variable: Output variable. A float array with shape
        of :math:`(B, N, L, 1)`. :math:`(N)` is the number of dimensions
        of word embedding.

    """
    e = embed(x)
    e = F.dropout(e, ratio=dropout)
    e = F.transpose(e, (0, 2, 1))
    e = e[:, :, :, None]
    return e


class RNNEncoder(chainer.Chain):

    def __init__(self, n_layers, n_vocab, embed_size, hidden_size, dropout=0.1):
        super(RNNEncoder, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, embed_size, ignore_label=-1,
                    initialW=embed_init)
            self.rnn = L.NStepLSTM(n_layers, embed_size, hidden_size, dropout)
        self.n_layers = n_layers
        self.output_size = hidden_size
        self.dropout = dropout

    def __call__(self, xs):
        exs = sequence_embed(self.embed, xs, self.dropout)
        last_h, last_c, ys = self.rnn(None, None, exs)
        assert(last_h.shape == (self.n_layers, len(xs), self.output_size))
        concat_outputs = last_h[-1]
        return concat_outputs


class CNNEncoder(chainer.Chain):

    def __init__(self, n_layers, n_vocab, embed_size, hidden_size, dropout=0.1):
        hidden_size /= 3
        super(CNNEncoder, self).__init__(
            embed=L.EmbedID(n_vocab, embed_size, ignore_label=-1,
                            initialW=embed_init),
            cnn_w3=L.Convolution2D(
                embed_size, hidden_size, ksize=(3, 1), stride=1, pad=(2, 0),
                nobias=True),
            cnn_w4=L.Convolution2D(
                embed_size, hidden_size, ksize=(4, 1), stride=1, pad=(3, 0),
                nobias=True),
            cnn_w5=L.Convolution2D(
                embed_size, hidden_size, ksize=(5, 1), stride=1, pad=(4, 0),
                nobias=True),
            mlp=MLP(n_layers, hidden_size * 3, dropout)
        )
        self.output_size = hidden_size * 3
        self.dropout = dropout

    def __call__(self, xs):
        x_block = chainer.dataset.convert.concat_examples(xs, padding=-1)
        ex_block = block_embed(self.embed, x_block, self.dropout)
        h_w3 = F.max(self.cnn_w3(ex_block), axis=2)
        h_w4 = F.max(self.cnn_w4(ex_block), axis=2)
        h_w5 = F.max(self.cnn_w5(ex_block), axis=2)
        h = F.concat([h_w3, h_w4, h_w5], axis=1)
        h = F.relu(h)
        h = F.dropout(h, ratio=self.dropout)
        h = self.mlp(h)
        return h


class DANEncoder(chainer.Chain):

    def __init__(self, n_vocab, embed_size, hidden_size, dropout):
        super(DANEncoder, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, embed_size, ignore_label=-1,
                    initialW=embed_init)
            self.linear = L.Linear(embed_size, hidden_size)
            self.batchnorm = L.BatchNormalization(hidden_size)
        self.dropout = dropout
        self.output_size = hidden_size
    
    def __call__(self, xs):
        x_block = chainer.dataset.convert.concat_examples(xs, padding=-1)
        ex_block = block_embed(self.embed, x_block)
        x_len = self.xp.array([len(x) for x in xs], 'i')[:, None, None]
        h = F.sum(ex_block, axis=2) / x_len

        h = self.linear(h)
        h = self.batchnorm(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=self.dropout)
        return h


class NNGuesser(chainer.Chain):

    def __init__(self, encoder, n_class, dropout):
        super(NNGuesser, self).__init__()
        with self.init_scope():
            self.encoder = encoder
            self.linear = L.Linear(encoder.output_size, n_class)
            self.batchnorm = L.BatchNormalization(n_class)
            # dropout
        self.dropout = dropout

    def load_glove(self, raw_path, vocab, size):
        print('Constructing embedding matrix')
        embed_w = np.random.uniform(-0.25, 0.25, size)
        with open(raw_path, 'r') as f:
            for line in tqdm(f):
                line = line.strip().split(" ")
                word = line[0]
                if word in vocab:
                    vec = np.array(line[1::], dtype=np.float32)
                    embed_w[vocab[word]] = vec
        embed_w = self.xp.array(embed_w, dtype=self.xp.float32)
        self.encoder.embed.W.data = embed_w
    
    def __call__(self, xs, ys):
        concat_outputs = self.predict(xs)
        concat_truths = F.concat(ys, axis=0)

        loss = F.softmax_cross_entropy(concat_outputs, concat_truths)
        accuracy = F.accuracy(concat_outputs, concat_truths)
        reporter.report({'loss': loss.data}, self)
        reporter.report({'accuracy': accuracy.data}, self)
        return loss

    def predict(self, xs, softmax=False, argmax=False):
        h = self.encoder(xs)

        h = self.linear(h)
        h = self.batchnorm(h)
        h = F.dropout(h, ratio=self.dropout)

        if softmax:
            return F.softmax(h).data
        elif argmax:
            return self.xp.argmax(h.data, axis=1)
        else:
            return h
