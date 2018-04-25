import datetime
import json
import os

import chainer
from chainer import training
from chainer.training import extensions

from qanta.buzzer.nets import RNNBuzzer
from qanta.buzzer.args import args
from qanta.buzzer.util import read_data, convert_seq, output_dir
from qanta.util.constants import BUZZER_TRAIN_FOLD, BUZZER_DEV_FOLD


def main():
    train = read_data(BUZZER_TRAIN_FOLD)
    valid = read_data(BUZZER_DEV_FOLD)
    print('# train data: {}'.format(len(train)))
    print('# valid data: {}'.format(len(valid)))

    train_iter = chainer.iterators.SerialIterator(train, args.batch_size)
    valid_iter = chainer.iterators.SerialIterator(
            valid, args.batch_size, repeat=False, shuffle=False)

    model = RNNBuzzer(args.n_input, args.n_layers, args.n_hidden,
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
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=output_dir)

    trainer.extend(extensions.Evaluator(
        valid_iter, model,
        converter=convert_seq, device=args.gpu))

    record_trigger = training.triggers.MaxValueTrigger(
        'validation/main/accuracy', (1, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        model, args.model_name),
        trigger=record_trigger)

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    trainer.extend(extensions.ProgressBar())

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
