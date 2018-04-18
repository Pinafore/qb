import os
import numpy as np
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer import cuda

from qanta import qlogging

from qanta.buzzer.progress import ProgressBar
from qanta.buzzer.util import load_quizbowl
from qanta.buzzer.iterator import QuestionIterator

log = qlogging.get(__name__)

class config:
    def __init__(self):
        self.n_hidden      = 200
        self.optimizer     = 'Adam'
        self.lr            = 1e-3
        self.max_grad_norm = 5
        self.batch_size    = 128
        self.model_dir     = 'output/buzzer/dqn_buzzer.npz'
        self.log_dir       = 'dqn_buzzer.log'


class MLP(chainer.Chain):

    def __init__(self, n_input, n_hidden, n_output):
        super(MLP, self).__init__(
                linear1=L.Linear(n_input, n_hidden),
                linear2=L.Linear(n_hidden, n_hidden),
                linear3=L.Linear(n_hidden, n_output))
    @property
    def xp(self):
        if not cuda.available or self.linear1._cpu:
            return np
        return cuda.cupy

    def get_device(self):
        if not cuda.available or self.linear1._cpu:
            return -1
        return self.linear1._device_id

    def __call__(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

class QTrainer(object):

    def __init__(self, model, train_iter, eval_iter, model_dir, log_dir):
        self.model = model
        self.train_iter = train_iter
        self.eval_iter = eval_iter
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.optimizer = chainer.optimizers.Adam(alpha=5 * 1e-4)
        self.optimizer.setup(self.model)
        self.optimizer.add_hook(chainer.optimizer.GradientClipping(5))

        self.r_correct = 10
        self.r_wrong = -5
        self.r_rush = -5
        self.r_late = -10

    def backprop(self, loss):
        self.optimizer.target.cleargrads()
        self.optimizer.update(lossfun=lambda: loss)

    def test(self, test_iter):
        '''returns a dictionary of buzzes'''
        device = self.model.get_device()
        buzzes = dict()
        for i in range(test_iter.size):
            batch = test_iter.next_batch(self.model.xp)
            length, batch_size, _ = batch.vecs.shape
            qvalues = [self.model(vec) for vec in batch.vecs] # length, batch, 2
            actions = F.argmax(F.stack(qvalues), axis=2).data # length, batch
            actions = actions.T.tolist()
            for q, a in zip(batch.qids, actions):
                q = q.tolist()
                buzzes[q] = -1 if not any(a) else a.index(1)
        return buzzes 
    
    def evaluate(self):

        def _do_one(inputs):
            t, action, result, length, hopeful, terminate = inputs
        
            if terminate or t >= length:
                return 0, 0, 0, 0, 0, 0, True

            reward, reward_hopeful = 0, 0
            buzz, correct, rush, late = 0, 0, 0, 0
            if action == 1:
                terminate = True
                buzz = 1
                if result == 1:
                    reward = 10
                    correct = 1
                else:
                    reward = -5
                    if hopeful:
                        rush = 1
                    #     reward -= 10
            elif t == length - 1:
                if hopeful:
                    late = 1
            #         reward = -10
            reward_hopeful = reward if hopeful else 0

            return reward, reward_hopeful, buzz, correct, rush, late, terminate

        progress_bar = ProgressBar(self.eval_iter.size, unit_iteration=True)
        epoch_stats = [0, 0, 0, 0, 0, 0]
        num_examples = 0
        num_hopeful = 0
        for i in range(self.eval_iter.size):
            batch = self.eval_iter.next_batch(self.model.xp)
            length, batch_size, _ = batch.vecs.shape
            qvalues = [self.model(vec) for vec in batch.vecs] # length * batch_size * 2
            actions = F.argmax(F.stack(qvalues), axis=2).data # length * batch_size
            hopeful = [any(x == 1) for x in batch.results.T] # batch_size
            terminate = [False for _ in range(batch_size)]
            num_examples += batch_size
            num_hopeful += sum(hopeful)

            for t in range(length):
                tt = [t for _ in range(batch_size)]
                inputs = zip(tt, actions[t], batch.results[t], batch.mask[t], hopeful, terminate)
                returns = map(_do_one, inputs)
                returns = list(map(list, zip(*returns)))
                terminate = returns[-1]
                returns = returns[:-1]
                epoch_stats = list(map(lambda x, y: x + sum(y), epoch_stats, returns))
                if all(terminate):
                    break
            # end of batch
            progress_bar(*self.eval_iter.epoch_detail)
        # end of epoch
        self.eval_iter.finalize(reset=True)
        progress_bar.finalize()
        epoch_stats[0] /= num_examples # reward / num_total
        epoch_stats[1] /= num_hopeful # reward_hopeful / num_hopeful
        epoch_stats = [num_examples, num_hopeful] + epoch_stats
        epoch_stats = BuzzStats(*epoch_stats)
        return epoch_stats

    def train_one_epoch(self, progress_bar, do_one=True):

        def _do_both(inputs):
            t, qvs, nqvs, result, mask, hopeful, terminate = inputs

            if terminate or mask != 1:
                return None, 0, 0, 0, 0, 0, 0, True

            action = int(F.argmax(qvs).data)
            # if action == 1:
            #     terminate = True
            
            reward = [0, 0]
            if result == 1:
                reward[1] = self.r_correct
            else:
                reward[1] = self.r_rush if hopeful else self.r_wrong
            if t == length - 1 and hopeful:
                reward[0] = self.r_late
            nqv = F.max(nqvs).data
            loss = F.square(reward[0] + 0.3 * nqv - qvs[0])
            loss += F.square(reward[1] - qvs[1])

            r = 0
            if action == 1:
                r = 10 if result == 1 else -5
            r_hope = r if hopeful else 0
            correct = int(action and result == 1)
            rush = 1 if (result != 1 and action and hopeful) else 0
            late = 1 if (t == length - 1 and not action and hopeful) else 0
            return loss, r, r_hope, action, correct, rush, late, terminate

        def _do_one(inputs):
            t, qvs, nqvs, result, mask, hopeful, terminate = inputs

            if terminate or mask != 1:
                return None, 0, 0, 0, 0, 0, 0, True

            action = int(F.argmax(qvs).data)
            if action == 1:
                terminate = True
            
            reward = 0
            if action == 1:
                if result == 1:
                    reward = self.r_correct
                else:
                    reward = self.r_rush if hopeful else self.r_wrong
            elif t == length - 1 and hopeful:
                reward = self.r_late
            
            qv = F.max(qvalues)
            nqv = F.max(nqvs).data
            if action == 1:
                loss = F.square(reward - qv)
            else:
                loss = F.square(reward + 0.5 * nqv - qv)

            r = 0
            if action == 1:
                r = 10 if result == 1 else -5
            r_hope = r if hopeful else 0
            correct = int(action and result == 1)
            rush = 1 if (result != 1 and action and hopeful) else 0
            late = 1 if (t == length - 1 and not action and hopeful) else 0
            return loss, r, r_hope, action, correct, rush, late, terminate

        epoch_stats = [0, 0, 0, 0, 0, 0]
        num_examples = 0
        num_hopeful = 0
        for i in range(self.train_iter.size):
            batch = self.train_iter.next_batch(self.model.xp)
            length, batch_size, _ = batch.vecs.shape
            hopeful = [any(x == 1) for x in batch.results.T]
            terminate = [False for _ in range(batch_size)]
            num_examples += batch_size
            num_hopeful += sum(hopeful)
            for t in range(length):
                qvalues = self.model(batch.vecs[t])
                if t + 1 < length:
                    nqvalues = self.model(batch.vecs[t + 1]) 
                else:
                    nqvalues = self.model.xp.zeros((batch_size, 2))

                tt = [t for _ in range(batch_size)]
                inputs = zip(tt, qvalues, nqvalues, batch.results[t],
                        batch.mask[t], hopeful, terminate)

                if do_one:
                    returns = map(_do_one, inputs)
                else:
                    returns = map(_do_both, inputs)

                returns = list(map(list, zip(*returns)))
                loss = returns[0]
                terminate = returns[-1]
                returns = returns[1:-1]
                loss = [x for x in loss if x is not None]
                if len(loss) > 0:
                    self.backprop(sum(loss))
                
                epoch_stats = list(map(lambda x, y: x + sum(y), epoch_stats, returns))
                if all(terminate):
                    break
            progress_bar(*self.train_iter.epoch_detail)
        # end of epoch
        self.train_iter.finalize()
        epoch_stats[0] /= num_examples # reward / num_total
        epoch_stats[1] /= num_hopeful # reward_hopeful / num_hopeful
        epoch_stats = [num_examples, num_hopeful] + epoch_stats
        epoch_stats = BuzzStats(*epoch_stats)
        return epoch_stats

    def run(self, n_epochs, train=True, evaluate=True, save_model=False):

        def get_output(output, stats):
            output += ' total ' + str(stats.num_total)
            output += ' ' + str(stats.num_hopeful)
            output += '  correct ' + str(stats.correct)
            output += '  buzz ' + str(stats.buzz)
            output += '  rush ' + str(stats.rush)
            output += '  late ' + str(stats.late)
            output += '  reward ' + '%.2f' % stats.reward
            output += '  ' + '%.2f' % stats.reward_hopeful
            return output

        with open(self.log_dir, 'a') as log_file:
            log_file.write(strftime("\n%Y-%m-%d %H:%M:%S\n", gmtime()))

        progress_bar = ProgressBar(n_epochs, unit_iteration=False)
        for epoch in range(n_epochs):
            if train:
                do_one = False
                train_stats = self.train_one_epoch(progress_bar, do_one=do_one)
                output = get_output('train', train_stats)
                with open(self.log_dir, 'a') as log_file:
                    log_file.write(output + '\n')
                log.info('[dqn]', output) 
                if save_model:
                    chainer.serializers.save_npz(self.model_dir, self.model)
                progress_bar.finalize()
                
            if evaluate:
                eval_stats = self.evaluate()
                output = get_output('eval ', eval_stats)
                with open(self.log_dir, 'a') as log_file:
                    log_file.write(output + '\n')
                log.info('[dqn]', output)
                progress_bar.finalize()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=-1)
    parser.add_argument('-l', '--load', action='store_true', default=False)
    parser.add_argument('-s', '--save', action='store_true', default=False)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-d', '--data_num', type=int, default=-1)
    return parser.parse_args()

def main():
    cfg = config()
    args = parse_args()

    option2id, all_guesses = load_quizbowl()
    train_iter = QuestionIterator(all_guesses['dev'], option2id, batch_size=cfg.batch_size,
            only_hopeful=ONLY_HOPEFUL)
    eval_iter = QuestionIterator(all_guesses['test'], option2id, batch_size=cfg.batch_size,
            only_hopeful=False)

    model = MLP(train_iter.n_input, 128, 2)

    if args.gpu != -1 and cuda.available:
        log.info('[dqn] using gpu', args.gpu)
        cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    if os.path.exists(cfg.model_dir) and args.load:
        log.info('[dqn] loading model')
        chainer.serializers.load_npz(cfg.model_dir, model)

    trainer = QTrainer(model, train_iter, eval_iter, cfg.model_dir, cfg.log_dir)
    trainer.run(20, train=True, evaluate=True, save_model=args.save)

if __name__ == '__main__':
    main()
