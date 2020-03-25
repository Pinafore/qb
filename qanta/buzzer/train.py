import os

import chainer
from chainer import training
from chainer.training import extensions

from qanta.buzzer.nets import RNNBuzzer, MLPBuzzer, LinearBuzzer
from qanta.buzzer.util import read_data, convert_seq
from qanta.util.constants import BUZZER_TRAIN_FOLD, BUZZER_DEV_FOLD


def main(model):
    train = read_data(fold=BUZZER_TRAIN_FOLD)
    valid = read_data(fold=BUZZER_DEV_FOLD)
    print('# train data: {}'.format(len(train)))
    print('# valid data: {}'.format(len(valid)))

    train_iter = chainer.iterators.SerialIterator(train, 64)
    valid_iter = chainer.iterators.SerialIterator(valid, 64, repeat=False, shuffle=False)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))

    updater = training.updaters.StandardUpdater(train_iter, optimizer, converter=convert_seq, device=0)
    trainer = training.Trainer(updater, (20, 'epoch'), out=model.model_dir)

    trainer.extend(extensions.Evaluator(valid_iter, model, converter=convert_seq, device=0))

    record_trigger = training.triggers.MaxValueTrigger('validation/main/accuracy', (1, 'epoch'))
    trainer.extend(extensions.snapshot_object(model, 'buzzer.npz'), trigger=record_trigger)

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.PrintReport([
        'epoch', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'elapsed_time'
    ]))

    if not os.path.isdir(model.model_dir):
        os.mkdir(model.model_dir)

    trainer.run()


if __name__ == '__main__':
    model = LinearBuzzer(n_input=22, n_layers=1, n_hidden=50, n_output=2, dropout=0.4)
    chainer.backends.cuda.get_device_from_id(0).use()
    model.to_gpu()
    main(model)
