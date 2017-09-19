import abc
from collections import defaultdict

import numpy as np


class Callback(abc.ABC):
    @abc.abstractmethod
    def on_epoch_end(self, logs):
        pass


class BaseLogger(Callback):
    def on_epoch_end(self, logs):
        print('Epoch {}: train_acc={:.4f} test_acc={:.4f} | train_loss={:.4f} test_loss={:.4f} | time={:.1f}'.format(
            len(logs['train_acc']),
            logs['train_acc'][-1], logs['test_acc'][-1],
            logs['train_loss'][-1], logs['test_loss'][-1],
            logs['train_time'][-1]
        ))
        return False, None

    def __repr__(self):
        return 'BaseLogger()'


class TerminateOnNaN(Callback):
    def on_epoch_end(self, logs):
        for key, arr in logs.items():
            if np.any(np.isnan(arr)):
                return True, 'NaN encountered in {} containing {}'.format(key, arr)
        else:
            return False, None

    def __repr__(self):
        return 'TerminateOnNaN()'


class EarlyStopping(Callback):
    def __init__(self, monitor='test_loss', min_delta=0, patience=1):
        if monitor.endswith('loss'):
            self.improvement_sign = 1
        elif monitor.endswith('accuracy'):
            self.improvement_sign = -1
        else:
            raise ValueError('Unrecognized monitor')
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.best_monitor_score = self.improvement_sign * float('inf')
        self.current_patience = patience

    def __repr__(self):
        return 'EarlyStopping(monitor={}, min_delta={}, patience={})'.format(
            self.monitor, self.min_delta, self.patience)

    def on_epoch_end(self, logs):
        if logs[self.monitor][-1] * self.improvement_sign < self.improvement_sign * self.best_monitor_score:
            self.current_patience = self.patience
            self.best_monitor_score = logs[self.monitor][-1]
        else:
            self.current_patience -= 1

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
    def __init__(self, save_function, filepath, monitor='test_loss', save_best_only=True):
        self.save_function = save_function
        self.filepath = filepath
        self.save_best_only = save_best_only
        if monitor.endswith('loss'):
            self.improvement_sign = 1
        elif monitor.endswith('accuracy'):
            self.improvement_sign = -1
        else:
            raise ValueError('Unrecognized monitor')
        self.monitor = monitor
        self.best_monitor_score = self.improvement_sign * float('inf')

    def on_epoch_end(self, logs):
        if logs[self.monitor][-1] * self.improvement_sign < self.improvement_sign * self.best_monitor_score:
            self.best_monitor_score = logs[self.monitor][-1]
            if self.save_best_only:
                self.save_function(self.filepath)
            else:
                self.save_function(self.filepath.format(epoch=len(logs['train_time']) - 1))
        return False, None


class TrainingManager:
    def __init__(self, callbacks):
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
            stop_training, reason = c.on_epoch_end(self.logs)
            if stop_training:
                callback_stop_reasons.append('{}: {}'.format(c.__name__, reason))

        if len(callback_stop_reasons) > 0:
            return True, callback_stop_reasons
        else:
            return False, []