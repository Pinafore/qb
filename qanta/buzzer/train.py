import datetime
import json
import os

import chainer
from chainer import training
from chainer.training import extensions

import nets
from args import args
from util import read_data, convert_seq


def main():
    train, valid = read_data()
    print('# train data: {}'.format(len(train)))
    print('# valid data: {}'.format(len(valid)))

    train_iter = chainer.iterators.SerialIterator(train, args.batch_size)
    valid_iter = chainer.iterators.SerialIterator(
            valid, args.batch_size, repeat=False, shuffle=False)
    
    model = nets.RNNBuzzer(args.n_input, args.n_layers, args.n_hidden,
            args.n_output, args.dropout)

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer,
        converter=convert_seq, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.outdir)

    trainer.extend(extensions.Evaluator(
        valid_iter, model,
        converter=convert_seq, device=args.gpu))

    record_trigger = training.triggers.MaxValueTrigger(
        'validation/main/accuracy', (1, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        model, 'best_model.npz'),
        trigger=record_trigger)

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    trainer.extend(extensions.ProgressBar())

    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
