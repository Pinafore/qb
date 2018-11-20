import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter


class RNNBuzzer(chainer.Chain):

    def __init__(self, n_input, n_layers, n_hidden, n_output, dropout=0.1):
        super(RNNBuzzer, self).__init__()
        with self.init_scope():
            self.encoder = L.NStepLSTM(n_layers, n_input, n_hidden, dropout)
            self.linear = L.Linear(n_hidden, n_output)
        self.n_layers = n_layers
        self.n_output = n_output
        self.dropout = dropout
        self.model_name = 'RNNBuzzer'
        self.model_dir = 'output/buzzer/RNNBuzzer'

    def __call__(self, xs, ys):
        concat_outputs = self.forward(xs)
        concat_outputs = F.softmax(concat_outputs, axis=1)
        concat_truths = F.concat(ys, axis=0)
        loss = F.softmax_cross_entropy(concat_outputs, concat_truths)
        accuracy = F.accuracy(concat_outputs, concat_truths)
        reporter.report({'loss': loss.data}, self)
        reporter.report({'accuracy': accuracy.data}, self)
        return loss

    def forward(self, xs):
        _, _, ys = self.encoder(None, None, xs)
        # ys is a list of hidden sequences
        ys = F.concat(ys, axis=0)   # batch_size * length, n_output
        ys = F.dropout(ys, self.dropout)
        ys = self.linear(ys)
        return ys

    def predict(self, xs, softmax=False, argmax=False):
        sections = np.cumsum(
                [len(x) for x in xs[:-1]], dtype=np.int32)
        ys = self.forward(xs)
        if softmax:
            ys = F.softmax(ys, axis=1).data
            ys = self.xp.split(ys, sections, axis=0)
        elif argmax:
            ys = self.xp.argmax(ys.data, axis=1)
            ys = self.xp.split(ys, sections, axis=0)
        else:
            ys = self.xp.split(ys.data, sections, axis=0)
        return ys


class MLPBuzzer(chainer.Chain):

    def __init__(self, n_input, n_layers, n_hidden, n_output, dropout=0.1):
        super(MLPBuzzer, self).__init__()
        input_layer = L.Linear(n_input, n_hidden)
        layers = chainer.Sequential(F.relu, L.Linear(n_hidden)).repeat(n_layers)
        layers.append(F.relu)
        layers.append(L.Linear(n_hidden, n_output))
        with self.init_scope():
            self.input_layer = input_layer
            self.layers = layers
        self.n_layers = n_layers
        self.n_output = n_output
        self.dropout = dropout
        self.model_name = 'MLPBuzzer'
        self.model_dir = 'output/buzzer/MLPBuzzer'

    def __call__(self, xs, ys):
        concat_outputs = self.forward(xs)
        concat_outputs = F.softmax(concat_outputs, axis=1)
        concat_truths = F.concat(ys, axis=0)
        loss = F.softmax_cross_entropy(concat_outputs, concat_truths)
        accuracy = F.accuracy(concat_outputs, concat_truths)
        reporter.report({'loss': loss.data}, self)
        reporter.report({'accuracy': accuracy.data}, self)
        return loss

    def forward(self, xs):
        xs = F.concat(xs, axis=0)
        ys = self.layers(self.input_layer(xs))
        return ys

    def predict(self, xs, softmax=False, argmax=False):
        sections = np.cumsum(
                [len(x) for x in xs[:-1]], dtype=np.int32)
        ys = self.forward(xs)
        if softmax:
            ys = F.softmax(ys, axis=1).data
            ys = self.xp.split(ys, sections, axis=0)
        elif argmax:
            ys = self.xp.argmax(ys.data, axis=1)
            ys = self.xp.split(ys, sections, axis=0)
        else:
            ys = self.xp.split(ys.data, sections, axis=0)
        return ys
