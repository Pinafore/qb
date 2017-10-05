import abc
from collections import defaultdict
from typing import List, Tuple, Optional

import numpy as np


class Callback(abc.ABC):
    @abc.abstractmethod
    def on_epoch_end(self, logs) -> Tuple[bool, Optional[str]]:
        pass


class BaseLogger(Callback):
    def __init__(self, log_func=print):
        self.log_func = log_func
    def on_epoch_end(self, logs):
        msg = 'Epoch {}: train_acc={:.4f} test_acc={:.4f} | train_loss={:.4f} test_loss={:.4f} | time={:.1f}'.format(
            len(logs['train_acc']),
            logs['train_acc'][-1], logs['test_acc'][-1],
            logs['train_loss'][-1], logs['test_loss'][-1],
            logs['train_time'][-1]
        )
        self.log_func(msg)

    def __repr__(self):
        return 'BaseLogger()'


class TerminateOnNaN(Callback):
    def on_epoch_end(self, logs):
        for key, arr in logs.items():
            if np.any(np.isnan(arr)):
                raise ValueError('NaN encountered')
        else:
            return False, None

    def __repr__(self):
        return 'TerminateOnNaN()'


class EarlyStopping(Callback):
    def __init__(self, monitor='test_loss', min_delta=0, patience=1, verbose=0, log_func=print):
        if monitor.endswith('loss'):
            self.improvement_sign = 1
        elif monitor.endswith('acc'):
            self.improvement_sign = -1
        else:
            raise ValueError('Unrecognized monitor')
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.best_monitor_score = self.improvement_sign * float('inf')
        self.current_patience = patience
        self.verbose = verbose
        self.log_func = log_func

    def __repr__(self):
        return 'EarlyStopping(monitor={}, min_delta={}, patience={})'.format(
            self.monitor, self.min_delta, self.patience)

    def on_epoch_end(self, logs):
        if logs[self.monitor][-1] * self.improvement_sign < self.improvement_sign * self.best_monitor_score:
            self.current_patience = self.patience
            self.best_monitor_score = logs[self.monitor][-1]
        else:
            self.current_patience -= 1
            if self.verbose > 0:
                self.log_func('Patience: reduced by one and waiting for {} epochs for improvement before stopping'.format(self.current_patience))

        if self.current_patience == 0:
            return True, 'Ran out of patience'
        else:
            return False, None


class MaxEpochStopping(Callback):
    def __init__(self, max_epochs):
        self.max_epochs = max_epochs

    def on_epoch_end(self, logs):
        if len(logs['train_time']) == self.max_epochs:
            return True, 'Max epochs reached'
        else:
            return False, None


class ModelCheckpoint(Callback):
    def __init__(self, save_function, filepath, monitor='test_loss', save_best_only=True, verbose=0, log_func=print):
        self.save_function = save_function
        self.filepath = filepath
        self.save_best_only = save_best_only
        if monitor.endswith('loss'):
            self.improvement_sign = 1
        elif monitor.endswith('acc'):
            self.improvement_sign = -1
        else:
            raise ValueError('Unrecognized monitor')
        self.monitor = monitor
        self.best_monitor_score = self.improvement_sign * float('inf')
        self.verbose = verbose
        self.log_func = log_func

    def on_epoch_end(self, logs):
        if logs[self.monitor][-1] * self.improvement_sign < self.improvement_sign * self.best_monitor_score:
            self.best_monitor_score = logs[self.monitor][-1]
            if self.save_best_only:
                if self.verbose > 0:
                    self.log_func('New best model, saving to: {}'.format(self.filepath))
                self.save_function(self.filepath)
            else:
                path = self.filepath.format(epoch=len(logs['train_time']) - 1)
                if self.verbose > 0:
                    self.log_func('New best model, saving to: {}'.format(path))
                self.save_function(path)


class Tensorboard(Callback):
    def __init__(self, experiment_name: str, log_dir=None):
        from pycrayon import CrayonClient
        self.client = CrayonClient(port=6007)
        self.experiment_name = experiment_name
        try:
            self.client.remove_experiment(experiment_name)
        except ValueError:
            pass
        self.experiment = self.client.create_experiment(experiment_name)

    def on_epoch_end(self, logs):
        self.experiment.add_scalar_value('train_loss', logs['train_loss'][-1])
        self.experiment.add_scalar_value('train_acc', logs['train_acc'][-1])
        self.experiment.add_scalar_value('test_loss', logs['test_loss'][-1])
        self.experiment.add_scalar_value('test_acc', logs['test_acc'][-1])
        self.experiment.add_scalar_value('train_time', logs['train_time'][-1])


class TrainingManager:
    def __init__(self, callbacks: List[Callback]):
        self.callbacks = callbacks
        self.logs = defaultdict(list)

    def instruct(self, train_time, train_loss, train_acc, test_time, test_loss, test_acc):
        self.logs['train_time'].append(train_time)
        self.logs['train_loss'].append(train_loss)
        self.logs['train_acc'].append(train_acc)
        self.logs['test_time'].append(test_time)
        self.logs['test_loss'].append(test_loss)
        self.logs['test_acc'].append(test_acc)

        callback_stop_reasons = []
        for c in self.callbacks:
            result = c.on_epoch_end(self.logs)
            if result is None:
                stop_training, reason = False, None
            else:
                stop_training, reason = result
            if stop_training:
                callback_stop_reasons.append('{}: {}'.format(c.__class__.__name__, reason))

        if len(callback_stop_reasons) > 0:
            return True, callback_stop_reasons
        else:
            return False, []
