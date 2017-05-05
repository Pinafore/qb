import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda

class MLP(chainer.ChainList):

    def __init__(self, n_input, n_hidden, n_output, n_layers, dropout=0):
        self.n_layers = n_layers + 2
        self.dropout = dropout
        layers = []
        layers.append(L.Linear(n_input, n_hidden))
        for i in range(n_layers):
            layers.append(L.Linear(n_hidden, n_hidden))
        layers.append(L.Linear(n_hidden, n_output))
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
            if self.dropout > 0:
                xs = F.dropout(xs, ratio=self.dropout, train=train)
            xs = self[i](xs)
        ys = F.log_softmax(xs)
        return ys


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
        return F.reshape(ys, (length * batch_size, -1))
