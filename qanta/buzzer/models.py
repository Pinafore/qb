import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda

class MLP(chainer.ChainList):

    def __init__(self, n_input, n_hidden, n_output, n_layers, dropout=0,
            batch_norm=False):
        self.dropout = dropout
        layers = []
        layers.append(L.Linear(n_input, n_hidden))
        for i in range(n_layers):
            if batch_norm:
                layers.append(L.BatchNormalization(size=n_hidden))
            layers.append(L.Linear(n_hidden, n_hidden))
        if batch_norm:
            layers.append(L.BatchNormalization(size=n_hidden))
        layers.append(L.Linear(n_hidden, n_output))
        if batch_norm:
            self.n_layers = n_layers * 2 + 3
        else:
            self.n_layers = n_layers + 2
        self.batch_norm = batch_norm
        super(MLP, self).__init__(*layers)

    @property
    def xp(self):
        if not cuda.available or self[0]._cpu:
            return np
        return cuda.cupy

    def get_device(self):
        if not cuda.available or self[0]._cpu:
            return -1
        return self[0]._device_id

    def __call__(self, xs, train=True):
        length, batch_size, _ = xs.shape
        xs = F.reshape(xs, (length * batch_size, -1))
        for i in range(self.n_layers):
            if not self.batch_norm and self.dropout > 0:
                xs = F.dropout(xs, ratio=self.dropout, train=train)
            xs = self[i](xs)
        return xs


class RNN(chainer.Chain):
    def __init__(self, n_input, n_hidden, n_output):
        super(RNN, self).__init__(
                rnn=L.LSTM(n_input, n_hidden),
                linear=L.Linear(n_hidden, n_output))

    @property
    def xp(self):
        if not cuda.available or self.linear._cpu:
            return np
        return cuda.cupy

    def get_device(self):
        if not cuda.available or self.linear._cpu:
            return -1
        return self.linear._device_id

    def __call__(self, xs, train=True):
        length, batch_size, _ = xs.shape
        self.rnn.reset_state()
        ys = F.stack([self.rnn(x) for x in xs], axis=0)
        ys = F.reshape(ys, (length * batch_size, -1))
        return ys
