import os
import json
import random
import argparse
import itertools
import numpy as np
import logging

import cupy
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer.dataset import concat_examples

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

answers_dir = 'answers.json'
vocabulary_dir = 'vocabulary.json'
train_questions_dir = 'train_questions.json'
dev_questions_dir = 'dev_questions.json'


class RNNModel(chainer.Chain):

    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(vocab_size, embed_size)
            self.rnn = L.LSTM(embed_size, hidden_size)
            self.linear = L.Linear(hidden_size, output_size)

        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

    def __call__(self, x):
        """
        Forward pass of a sentence.
        :param x: a batch of sentences
        :return h: final hidden states
        """
        xs = self.embed(x)
        xs = F.swapaxes(xs, 0, 1) # time, batch, embed
        self.rnn.reset_state()
        for x in xs:
            h = F.dropout(self.rnn(x))
        h = self.linear(h)
        return h


def load_embeddings(path, vocab_size, embed_size):
    emb = np.zeros((vocab_size, embed_size), dtype=np.float32)
    size = os.stat(path).st_size
    with open(path, 'rb') as f:
        pos = 0
        idx = 0
        while pos < size:
            chunk = np.load(f)
            chunk_size = chunk.shape[0]
            emb[idx : idx+chunk_size, :] = chunk
            idx += chunk_size
            pos = f.tell()
    return emb


class BPTTUpdater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer, bprop_len, device):
        super(BPTTUpdater, self).__init__(
            train_iter, optimizer, device=device)
        self.bprop_len = bprop_len

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        loss = 0
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        batch = train_iter.__next__()
        x, t = concat_examples(batch, self.device, 0)
        loss = optimizer.target(chainer.Variable(x), chainer.Variable(t))

        optimizer.target.cleargrads()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=128,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--bproplen', '-l', type=int, default=35,
                        help='Number of words in each mini-batch '
                             '(= length of truncated BPTT)')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gradclip', '-c', type=float, default=5,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.set_defaults(test=False)
    parser.add_argument('--hidden_size', type=int, default=300,
                        help='Number of LSTM units in each layer')
    parser.add_argument('--model', '-m', default='model.npz',
                        help='Model file name to serialize')
    args = parser.parse_args()

    logger.info('loading answers..')
    with open(answers_dir, 'r') as f:
        checkpoint = json.loads(f.read())
        id_to_answer = checkpoint['id_to_answer']
        answer_to_id = checkpoint['answer_to_id']
    logger.info('number of answers: {}'.format(len(id_to_answer)))
    
    logger.info('loading vocabulary..')
    with open(vocabulary_dir, 'r') as f:
        checkpoint = json.loads(f.read())
        id_to_word = checkpoint['id_to_word']
        word_to_id = checkpoint['word_to_id']
    logger.info('vocabulary size: {}'.format(len(id_to_word)))

    logger.info('loading questions..')
    with open(train_questions_dir, 'r') as f:
        train_questions = json.loads(f.read())
    with open(dev_questions_dir, 'r') as f:
        dev_questions = json.loads(f.read())
    logger.info('number of training questions: {}'.format(len(train_questions)))
    logger.info('number of dev questions: {}'.format(len(dev_questions)))

    def convert_dataset(questions):
        if isinstance(questions, dict):
            questions = list(questions.values())
    
        sentences = []
        labels = []
        for q in questions:
            a = q['answer']
            a = answer_to_id[a] if isinstance(a, str) else a
            for sent in q['sentences']:
                if isinstance(sent, list):
                    sent = np.array(sent, dtype=np.int32)
                sentences.append(sent)
                labels.append(a)
        return list(zip(sentences, labels))
    
    train_dataset = convert_dataset(train_questions)
    dev_dataset = convert_dataset(dev_questions)

    train_iter = chainer.iterators.SerialIterator(train_dataset, args.batch_size)
    dev_iter = chainer.iterators.SerialIterator(dev_dataset, args.batch_size,
            repeat=False)

    rnn = RNNModel(len(word_to_id), args.hidden_size,
            args.hidden_size, len(answer_to_id))
    model = L.Classifier(rnn)

    # if args.embeddings:
    #     model.embed.W.data = load_embeddings(
    #             args.embeddings, args.vocab_size, args.embed_size)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()
        # model.embed.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

    updater = BPTTUpdater(train_iter, optimizer, args.bproplen, args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    eval_model = model.copy()
    eval_rnn = eval_model.predictor
    trainer.extend(extensions.Evaluator(
            dev_iter, eval_model, device=args.gpu))

    interval = 10 if args.test else 500
    trainer.extend(extensions.LogReport(trigger=(interval, 'iteration')))
    trainer.extend(extensions.PrintReport(
                ['epoch', 'iteration', 'accuracy']),
            trigger=(interval, 'iteration'))
    trainer.extend(extensions.ProgressBar(
            update_interval=1 if args.test else 10))
    trainer.extend(extensions.snapshot())
    trainer.extend(extensions.snapshot_object(
            model, 'model_iter_{.updater.iteration}'))
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()
    
if __name__ == '__main__':
    main()
