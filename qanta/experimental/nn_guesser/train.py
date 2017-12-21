import os
import json
import numpy as np
import argparse
import datetime

import chainer
from chainer import training
from chainer.training import extensions

import nets
from nlp_utils import convert_seq
import dataset

def main():
    current_datetime = '{}'.format(datetime.datetime.today())
    parser = argparse.ArgumentParser(
        description='Chainer NN guesser.')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=30,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result/nn_guesser',
                        help='Directory to output the result')
    parser.add_argument('--unit', '-u', type=int, default=300,
                        help='Number of units')
    parser.add_argument('--layer', '-l', type=int, default=1,
                        help='Number of layers of RNN or MLP following CNN')
    parser.add_argument('--dropout', '-d', type=float, default=0.4,
                        help='Dropout rate')
    parser.add_argument('--model', '-model', default='bow',
                        choices=['cnn', 'rnn', 'bow'],
                        help='Name of encoder model type.')
    parser.add_argument('--resume', '-resume', action='store_true',
                        help='Resume training.')
    parser.add_argument('--glove', default='data/glove.6B.300d.txt',
                        help='Path to glove embedding file.')
    parser.set_defaults(resume=False)

    args = parser.parse_args()

    if args.resume:
        with open(os.path.join(args.out, 'args.json')) as f:
            args.__dict__ = json.loads(f.read())
        args.resume = True
    print(json.dumps(args.__dict__, indent=2))

    train, dev, vocab, answers = dataset.get_quizbowl(
            data_dir='data/nn_guesser',
            split_sentences=True)

    print('# train data: {}'.format(len(train)))
    print('# dev data: {}'.format(len(dev)))
    print('# vocab: {}'.format(len(vocab)))
    n_class = len(set([int(d[1]) for d in train]))
    print('# class: {}'.format(n_class))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    dev_iter = chainer.iterators.SerialIterator(dev, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Setup a model
    if args.model == 'rnn':
        Encoder = nets.RNNEncoder
    elif args.model == 'cnn':
        Encoder = nets.CNNEncoder
    elif args.model == 'bow':
        Encoder = nets.BOWMLPEncoder
    encoder = Encoder(n_layers=args.layer, n_vocab=len(vocab),
                      n_units=args.unit, dropout=args.dropout)

    model = nets.TextClassifier(encoder, n_class)

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))

    # Set up a trainer
    updater = training.StandardUpdater(
        train_iter, optimizer,
        converter=convert_seq, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the dev dataset for each epoch
    trainer.extend(extensions.Evaluator(
        dev_iter, model, converter=convert_seq, device=args.gpu))

    # Take a best snapshot
    record_trigger = training.triggers.MaxValueTrigger(
        'validation/main/accuracy', (1, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        model, 'best_model.npz'),
        trigger=record_trigger)

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    # Save vocabulary and model's setting
    if not os.path.isdir(args.out):
        os.mkdir(args.out)
    # current = os.path.dirname(os.path.abspath(__file__))
    vocab_path = os.path.join(args.out, 'vocab.json')
    answers_path = os.path.join(args.out, 'answers.json')
    embed_path = os.path.join(args.out, 'embed_w.npz')
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)
    with open(answers_path, 'w') as f:
        json.dump(answers, f)
    model_path = os.path.join(args.out, 'best_model.npz')
    model_setup = args.__dict__
    model_setup['vocab_path'] = vocab_path
    model_setup['answers_path'] = answers_path
    model_setup['model_path'] = model_path
    model_setup['n_class'] = n_class
    model_setup['datetime'] = current_datetime
    with open(os.path.join(args.out, 'args.json'), 'w') as f:
        json.dump(model_setup, f)

    if args.resume:
        print('loading model {}'.format(model_path))
        chainer.serializers.load_npz(model_path, model)
    else:
        if os.path.isfile(args.glove):
            model.load_glove(embed_path, args.glove, vocab)

    if args.model == 'bow':
        print(model.encoder.bow_encoder.embed.W)
    else:
        print(model.encoder.embed.W)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
