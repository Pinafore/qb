import os
import json
import random
import argparse
import itertools
import numpy as np
import logging
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")

import cupy
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer.dataset import concat_examples

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

answers_dir = "answers.json"
vocabulary_dir = "vocabulary.json"
train_questions_dir = "train_questions.json"
dev_questions_dir = "dev_questions.json"


class RNNModel(chainer.Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(vocab_size, embed_size)
            self.rnn = L.LSTM(embed_size, hidden_size)
            self.linear = L.Linear(hidden_size, output_size)

    def __call__(self, xs):
        """
        Forward pass of a sentence.
        :param xs: a batch of sentences
        :return h: final hidden states
        """
        xs = self.embed(xs)
        xs = F.swapaxes(xs, 0, 1)  # time, batch, embed
        self.rnn.reset_state()
        for x in xs:
            h = self.rnn(x)
        h = F.tanh(self.linear(h))
        return h


class DANModel(chainer.Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(DANModel, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(vocab_size, embed_size)
            self.linear1 = L.Linear(embed_size, hidden_size)
            self.linear2 = L.Linear(hidden_size, output_size)

    def __call__(self, xs):
        xs = self.embed(xs)
        batch_size, length, _ = xs.shape
        h = F.sum(xs, axis=1) / length
        h = F.tanh(self.linear1(h))
        h = F.tanh(self.linear2(h))
        return h


def load_glove(glove_path, word_to_id, embed_size):
    vocab_size = len(word_to_id)
    embed_W = np.zeros((vocab_size, embed_size), dtype=np.float32)
    with open(glove_path, "r") as fi:
        logger.info("loading glove vectors..")
        for line in tqdm(fi):
            line = line.strip().split(" ")
            word = line[0]
            if word in word_to_id:
                vec = np.array(line[1::], dtype=np.float32)
                embed_W[word_to_id[word]] = vec
    return embed_W


def converter(batch, device):
    x, t = concat_examples(batch, device, 0)
    return x, t


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=32,
        help="Number of examples in each mini-batch",
    )
    parser.add_argument(
        "--bproplen",
        "-l",
        type=int,
        default=35,
        help="Number of words in each mini-batch " "(= length of truncated BPTT)",
    )
    parser.add_argument(
        "--epoch",
        "-e",
        type=int,
        default=20,
        help="Number of sweeps over the dataset to train",
    )
    parser.add_argument(
        "--gpu", "-g", type=int, default=0, help="GPU ID (negative value indicates CPU)"
    )
    parser.add_argument(
        "--gradclip",
        "-c",
        type=float,
        default=5,
        help="Gradient norm threshold to clip",
    )
    parser.add_argument(
        "--out", "-o", default="result", help="Directory to output the result"
    )
    parser.add_argument(
        "--resume", "-r", default="", help="Resume the training from snapshot"
    )
    parser.add_argument(
        "--test", action="store_true", help="Use tiny datasets for quick tests"
    )
    parser.set_defaults(test=False)
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=300,
        help="Number of LSTM units in each layer",
    )
    parser.add_argument(
        "--embed_size", type=int, default=300, help="Size of embeddings"
    )
    parser.add_argument(
        "--model", "-m", default="model.npz", help="Model file name to serialize"
    )
    parser.add_argument(
        "--glove",
        default="data/glove.6B.300d.txt",
        help="Path to glove embedding file.",
    )
    args = parser.parse_args()
    return args


def main1():
    args = parse_args()

    logger.info("loading answers..")
    with open(answers_dir, "r") as f:
        checkpoint = json.loads(f.read())
        id_to_answer = checkpoint["id_to_answer"]
        answer_to_id = checkpoint["answer_to_id"]
    logger.info("number of answers: {}".format(len(id_to_answer)))

    logger.info("loading vocabulary..")
    with open(vocabulary_dir, "r") as f:
        checkpoint = json.loads(f.read())
        id_to_word = checkpoint["id_to_word"]
        word_to_id = checkpoint["word_to_id"]
    logger.info("vocabulary size: {}".format(len(id_to_word)))

    logger.info("loading questions..")
    with open(train_questions_dir, "r") as f:
        train_questions = json.loads(f.read())
    with open(dev_questions_dir, "r") as f:
        dev_questions = json.loads(f.read())
    logger.info("number of training questions: {}".format(len(train_questions)))
    logger.info("number of dev questions: {}".format(len(dev_questions)))

    def convert_dataset(questions):
        if isinstance(questions, dict):
            questions = list(questions.values())

        sentences = []
        labels = []
        for q in questions:
            a = q["answer"]
            a = answer_to_id[a] if isinstance(a, str) else a
            for sent in q["sentences"]:
                if isinstance(sent, list):
                    sent = np.array(sent, dtype=np.int32)
                sentences.append(sent)
                labels.append(a)
        return list(zip(sentences, labels))

    train_dataset = convert_dataset(train_questions)
    dev_dataset = convert_dataset(dev_questions)

    train_iter = chainer.iterators.SerialIterator(train_dataset, args.batch_size)
    dev_iter = chainer.iterators.SerialIterator(
        dev_dataset, args.batch_size, repeat=False
    )

    vocab_size = len(word_to_id)
    output_size = len(answer_to_id)
    model = DANModel(vocab_size, args.embed_size, args.hidden_size, output_size)

    # if os.path.isfile(args.glove):
    #     rnn.embed.W.data = load_glove(
    #             args.glove, word_to_id, args.embed_size)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()
        # model.predictor.embed.to_gpu()

    optimizer = chainer.optimizers.SGD(lr=1.0)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

    iteration = 0
    sum_loss = 0
    sum_acc = 0
    count = 0
    while train_iter.epoch < args.epoch:
        iteration += 1
        count += 1
        batch = train_iter.__next__()
        x, t = converter(batch, args.gpu)
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        sum_loss += loss.data
        sum_acc += F.accuracy(y, t).data
        optimizer.target.cleargrads()
        loss.backward()
        optimizer.update()

        if iteration % 10 == 0:
            print("{}: {} {}".format(iteration, sum_loss / count, sum_acc / count))
            sum_loss = 0
            sum_acc = 0
            count = 0

        if train_iter.is_new_epoch:
            print("epoch: ", train_iter.epoch)


def main():
    args = parse_args()

    logger.info("loading answers..")
    with open(answers_dir, "r") as f:
        checkpoint = json.loads(f.read())
        id_to_answer = checkpoint["id_to_answer"]
        answer_to_id = checkpoint["answer_to_id"]
    logger.info("number of answers: {}".format(len(id_to_answer)))

    logger.info("loading vocabulary..")
    with open(vocabulary_dir, "r") as f:
        checkpoint = json.loads(f.read())
        id_to_word = checkpoint["id_to_word"]
        word_to_id = checkpoint["word_to_id"]
    logger.info("vocabulary size: {}".format(len(id_to_word)))

    logger.info("loading questions..")
    with open(train_questions_dir, "r") as f:
        train_questions = json.loads(f.read())
    with open(dev_questions_dir, "r") as f:
        dev_questions = json.loads(f.read())
    logger.info("number of training questions: {}".format(len(train_questions)))
    logger.info("number of dev questions: {}".format(len(dev_questions)))

    def convert_dataset(questions):
        if isinstance(questions, dict):
            questions = list(questions.values())

        sentences = []
        labels = []
        for q in questions:
            a = q["answer"]
            a = answer_to_id[a] if isinstance(a, str) else a
            for sent in q["sentences"]:
                if isinstance(sent, list):
                    sent = np.array(sent, dtype=np.int32)
                sentences.append(sent)
                labels.append(a)
        return list(zip(sentences, labels))

    train_dataset = convert_dataset(train_questions)
    dev_dataset = convert_dataset(dev_questions)

    train_iter = chainer.iterators.SerialIterator(train_dataset, args.batch_size)
    dev_iter = chainer.iterators.SerialIterator(
        dev_dataset, args.batch_size, repeat=False
    )

    vocab_size = len(word_to_id)
    output_size = len(answer_to_id)
    rnn = DANModel(vocab_size, args.embed_size, args.hidden_size, output_size)

    # if os.path.isfile(args.glove):
    #     rnn.embed.W.data = load_glove(
    #             args.glove, word_to_id, args.embed_size)

    model = L.Classifier(rnn)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()
        # model.predictor.embed.to_gpu()

    optimizer = chainer.optimizers.SGD(lr=1.0)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

    updater = training.StandardUpdater(
        train_iter, optimizer, converter=converter, device=args.gpu
    )
    trainer = training.Trainer(updater, (args.epoch, "epoch"), out=args.out)

    eval_model = model.copy()
    eval_rnn = eval_model.predictor
    trainer.extend(
        extensions.Evaluator(dev_iter, eval_model, converter=converter, device=args.gpu)
    )

    interval = 10 if args.test else 100
    trainer.extend(extensions.LogReport(trigger=(interval, "iteration")))
    trainer.extend(
        extensions.PrintReport(
            [
                "epoch",
                "main/loss",
                "main/accuracy",
                "validation/main/loss",
                "validation/main/accuracy",
                "elapsed_time",
            ]
        ),
        trigger=(interval, "iteration"),
    )
    # trainer.extend(extensions.PlotReport([
    #     'main/loss', 'validation/main/loss'],
    #     x_key='epoch', file_name='loss.png'))
    # trainer.extend(extensions.PlotReport([
    #     'main/accuracy', 'validation/main/accuracy'],
    #     x_key='epoch', file_name='accuracy.png'))
    # trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    # trainer.extend(extensions.snapshot())
    # trainer.extend(extensions.snapshot_object(
    #         model, 'model_iter_{.updater.iteration}'))
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()
    chainer.serializers.save_npz(args.model, model)


if __name__ == "__main__":
    main1()
