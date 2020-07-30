import os
import json
import numpy as np
import argparse
import datetime

import chainer
from chainer import training
from chainer.training import extensions

from qanta.preprocess import preprocess_dataset
from qanta.datasets.quiz_bowl import QuizBowlDataset

from qanta.experimental.nn_guesser import nets
from qanta.experimental.nn_guesser.nlp_utils import convert_seq, transform_to_array


def get_quizbowl():
    qb_dataset = QuizBowlDataset(guesser_train=True, buzzer_train=False)
    training_data = qb_dataset.training_data()
    (
        train_x,
        train_y,
        dev_x,
        dev_y,
        i_to_word,
        class_to_i,
        i_to_class,
    ) = preprocess_dataset(training_data)
    i_to_word = ["<unk>", "<eos>"] + sorted(i_to_word)
    word_to_i = {x: i for i, x in enumerate(i_to_word)}
    train = transform_to_array(zip(train_x, train_y), word_to_i)
    dev = transform_to_array(zip(dev_x, dev_y), word_to_i)
    return train, dev, word_to_i, i_to_class


def main():
    current_datetime = "{}".format(datetime.datetime.today())
    parser = argparse.ArgumentParser(description="Chainer NN guesser.")
    parser.add_argument(
        "--batchsize",
        type=int,
        default=64,
        help="Number of examples in each mini-batch",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="Number of sweeps over the dataset to train",
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU ID (negative value indicates CPU)"
    )
    parser.add_argument(
        "--out", default="result/nn_guesser", help="Directory to output the result"
    )
    parser.add_argument(
        "--model",
        default="dan",
        choices=["cnn", "rnn", "dan"],
        help="Name of encoder model type.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume training.")
    parser.add_argument(
        "--glove",
        default="data/external/deep/glove.6B.300d.txt",
        help="Path to glove embedding file.",
    )
    parser.set_defaults(resume=False)
    args = parser.parse_args()

    if args.resume:
        with open(os.path.join(args.out, "args.json")) as f:
            args.__dict__ = json.loads(f.read())
        args.resume = True
    print(json.dumps(args.__dict__, indent=2))

    train, dev, vocab, answers = get_quizbowl()

    n_vocab = len(vocab)
    n_class = len(set([int(d[1]) for d in train]))
    embed_size = 300
    hidden_size = 512
    hidden_dropout = 0.3
    output_dropout = 0.2
    gradient_clipping = 0.25

    print("# train data: {}".format(len(train)))
    print("# dev data: {}".format(len(dev)))
    print("# vocab: {}".format(len(vocab)))
    print("# class: {}".format(n_class))
    print("embedding size: {}".format(embed_size))
    print("hidden size: {}".format(hidden_size))
    print("hidden dropout: {}".format(hidden_dropout))
    print("output dropout: {}".format(output_dropout))
    print("gradient clipping: {}".format(gradient_clipping))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    dev_iter = chainer.iterators.SerialIterator(
        dev, args.batchsize, repeat=False, shuffle=False
    )

    # Setup a model
    if args.model == "dan":
        encoder = nets.DANEncoder(
            n_vocab, embed_size, hidden_size, dropout=hidden_dropout
        )
    elif args.model == "rnn":
        encoder = nets.RNNEncoder(1, n_vocab, embed_size, hidden_size)
    model = nets.NNGuesser(encoder, n_class, dropout=output_dropout)

    if not args.resume:
        model.load_glove(args.glove, vocab, (n_vocab, embed_size))

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam(alpha=0.001)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))
    optimizer.add_hook(chainer.optimizer.GradientClipping(gradient_clipping))

    # Set up a trainer
    updater = training.StandardUpdater(
        train_iter, optimizer, converter=convert_seq, device=args.gpu
    )
    trainer = training.Trainer(updater, (args.epoch, "epoch"), out=args.out)

    # Evaluate the model with the dev dataset for each epoch
    trainer.extend(
        extensions.Evaluator(dev_iter, model, converter=convert_seq, device=args.gpu)
    )

    # Take a best snapshot
    record_trigger = training.triggers.MaxValueTrigger(
        "validation/main/accuracy", (1, "epoch")
    )
    trainer.extend(
        extensions.snapshot_object(model, "best_model.npz"), trigger=record_trigger
    )

    # Exponential decay of learning rate
    # trainer.extend(extensions.ExponentialShift('alpha', 0.5))

    # Write a log of evaluation statistics for each epoch

    trainer.extend(extensions.LogReport())
    trainer.extend(
        extensions.PrintReport(
            [
                "epoch",
                "main/loss",
                "validation/main/loss",
                "main/accuracy",
                "validation/main/accuracy",
                "elapsed_time",
            ]
        )
    )

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    # Save vocabulary and model's setting
    if not os.path.isdir(args.out):
        os.mkdir(args.out)
    # current = os.path.dirname(os.path.abspath(__file__))
    vocab_path = os.path.join(args.out, "vocab.json")
    answers_path = os.path.join(args.out, "answers.json")
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)
    with open(answers_path, "w") as f:
        json.dump(answers, f)
    model_path = os.path.join(args.out, "best_model.npz")
    model_setup = args.__dict__
    model_setup["vocab_path"] = vocab_path
    model_setup["answers_path"] = answers_path
    model_setup["model_path"] = model_path
    model_setup["n_class"] = n_class
    model_setup["datetime"] = current_datetime
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(model_setup, f)

    if args.resume:
        print("loading model {}".format(model_path))
        chainer.serializers.load_npz(model_path, model)

    # Run the training
    trainer.run()


if __name__ == "__main__":
    main()
