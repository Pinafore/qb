"""An example of training DQN against OpenAI Gym Envs.

This script is an example of training a DQN agent against OpenAI Gym envs.
Both discrete and continuous action spaces are supported. For continuous action
spaces, A NAF (Normalized Advantage Function) is used to approximate Q-values.
"""
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import argparse
import sys
import pickle
import numpy as np
import gym
import gym.wrappers
from gym import spaces
gym.undo_logger_setup()

from chainer import optimizers
from chainer import serializers
import chainer.functions as F
from chainerrl.agents.dqn import DQN
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import misc
from chainerrl import q_functions
from chainerrl import replay_buffer

from qanta.util import constants as c
from qanta.buzzer.util import load_quizbowl
from qanta.buzzer.iterator import QuestionIterator
from qanta.buzzer.rnn_0 import dense_vector
from qanta.buzzer.buzzing_env import BuzzingGame
from qanta.buzzer.report import report
from qanta.buzzer.trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', type=str, default='dqn_out')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--final-exploration-steps',
                    type=int, default=10 ** 4)
parser.add_argument('--start-epsilon', type=float, default=1.0)
parser.add_argument('--end-epsilon', type=float, default=0.1)
parser.add_argument('--demo', action='store_true', default=False)
parser.add_argument('--load', type=str, default=None)
parser.add_argument('--steps', type=int, default=10 ** 5)
parser.add_argument('--prioritized-replay', action='store_true')
parser.add_argument('--episodic-replay', action='store_true')
parser.add_argument('--replay-start-size', type=int, default=None)
parser.add_argument('--target-update-interval', type=int, default=10 ** 2)
parser.add_argument('--target-update-method', type=str, default='hard')
parser.add_argument('--soft-update-tau', type=float, default=1e-2)
parser.add_argument('--update-interval', type=int, default=1)
parser.add_argument('--eval-n-runs', type=int, default=100)
parser.add_argument('--eval-interval', type=int, default=10 ** 4)
parser.add_argument('--n-hidden-channels', type=int, default=100)
parser.add_argument('--n-hidden-layers', type=int, default=2)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--minibatch-size', type=int, default=64)
parser.add_argument('--render-train', action='store_true')
parser.add_argument('--render-eval', action='store_true')
parser.add_argument('--monitor', action='store_true')
parser.add_argument('--reward-scale-factor', type=float, default=1e-3)


def get_buzzes(model, test_iter):
    buzzes = dict()
    # progress_bar = ProgressBar(test_iter.size, unit_iteration=True)
    for i in range(test_iter.size):
        batch = test_iter.next_batch(np)
        length, batch_size, _ = batch.vecs.shape
        ys = [model(x).q_values for x in batch.vecs] # length * (batch_size, n_actions)
        ys = F.softmax(F.concat(ys, axis=0)) # length * batch_size, n_actions 
        ys = F.swapaxes(F.reshape(ys, (length, batch_size, -1)), 0, 1)
        ys.to_cpu()
        masks = batch.mask.T.tolist()
        assert len(masks) == batch_size
        for qnum, scores, mask in zip(batch.qids, ys.data, masks):
            if isinstance(qnum, np.ndarray):
                qnum = qnum.tolist()
            total = int(sum(mask))
            buzzes[qnum] = scores[:total].tolist()

            # for t in range(total):
            #     q = buzzes[qnum][t][0]
            #     if q < 0.6 and q > 0.5:
            #         buzzes[qnum][t][0] -= 0.1
            #         buzzes[qnum][t][1] += 0.1
                        
        # progress_bar(*test_iter.epoch_detail)
    test_iter.finalize(reset=True)
    # progress_bar.finalize()
    return buzzes

def main():
    import logging
    logging.basicConfig(level=logging.WARNING)

    args = parser.parse_args()

    args.outdir = experiments.prepare_output_dir(
        args, args.outdir, argv=sys.argv)
    print('Output files are saved in {}'.format(args.outdir))

    if args.seed is not None:
        misc.set_random_seed(args.seed)

    option2id, all_guesses = load_quizbowl()
    train_iter = QuestionIterator(all_guesses[c.BUZZER_DEV_FOLD], option2id,
            batch_size=1, make_vector=dense_vector)

    env = BuzzingGame(train_iter)

    timestep_limit = 300
    obs_size = env.observation_size
    action_space = env.action_space

    n_actions = action_space.n
    q_func = q_functions.FCStateQFunctionWithDiscreteAction(
        obs_size, n_actions,
        n_hidden_channels=args.n_hidden_channels,
        n_hidden_layers=args.n_hidden_layers)
    # Use epsilon-greedy for exploration
    explorer = explorers.LinearDecayEpsilonGreedy(
        args.start_epsilon, args.end_epsilon, args.final_exploration_steps,
        action_space.sample)

    opt = optimizers.Adam()
    opt.setup(q_func)

    rbuf_capacity = 5 * 10 ** 5
    if args.episodic_replay:
        if args.minibatch_size is None:
            args.minibatch_size = 4
        if args.replay_start_size is None:
            args.replay_start_size = 10
        if args.prioritized_replay:
            betasteps = \
                (args.steps - timestep_limit * args.replay_start_size) \
                // args.update_interval
            rbuf = replay_buffer.PrioritizedEpisodicReplayBuffer(
                rbuf_capacity, betasteps=betasteps)
        else:
            rbuf = replay_buffer.EpisodicReplayBuffer(rbuf_capacity)
    else:
        if args.minibatch_size is None:
            args.minibatch_size = 32
        if args.replay_start_size is None:
            args.replay_start_size = 1000
        if args.prioritized_replay:
            betasteps = (args.steps - args.replay_start_size) \
                // args.update_interval
            rbuf = replay_buffer.PrioritizedReplayBuffer(
                rbuf_capacity, betasteps=betasteps)
        else:
            rbuf = replay_buffer.ReplayBuffer(rbuf_capacity)

    def phi(obs):
        return obs.astype(np.float32)

    agent = DQN(q_func, opt, rbuf, gpu=args.gpu, gamma=args.gamma,
                explorer=explorer, replay_start_size=args.replay_start_size,
                target_update_interval=args.target_update_interval,
                update_interval=args.update_interval,
                phi=phi, minibatch_size=args.minibatch_size,
                target_update_method=args.target_update_method,
                soft_update_tau=args.soft_update_tau,
                episodic_update=args.episodic_replay, episodic_update_len=16)

    if args.load:
        agent.load(args.load)

    eval_env = BuzzingGame(train_iter)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_runs=args.eval_n_runs,
            max_episode_len=timestep_limit)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        experiments.train_agent_with_evaluation(
            agent=agent, env=env, steps=args.steps,
            eval_n_runs=args.eval_n_runs, eval_interval=args.eval_interval,
            outdir=args.outdir, eval_env=eval_env,
            max_episode_len=timestep_limit)

    serializers.save_npz('dqn.npz', q_func)

    dev_iter = QuestionIterator(all_guesses[c.BUZZER_DEV_FOLD], option2id,
            batch_size=128, make_vector=dense_vector)
    dev_buzzes = get_buzzes(q_func, dev_iter)
    dev_buzzes_dir = 'output/buzzer/rl/dev_buzzes.pkl'
    with open(dev_buzzes_dir, 'wb') as f:
        pickle.dump(dev_buzzes, f)
    print('Dev buzz {} saved to {}'.format(len(dev_buzzes), dev_buzzes_dir))

    report(dev_buzzes_dir)

def test():
    args = parser.parse_args()
    option2id, all_guesses = load_quizbowl()
    dev_iter = QuestionIterator(all_guesses[c.BUZZER_DEV_FOLD], option2id,
            batch_size=128, make_vector=dense_vector)
    env = BuzzingGame(dev_iter)

    obs_size = env.observation_size
    action_space = env.action_space

    n_actions = action_space.n
    q_func = q_functions.FCStateQFunctionWithDiscreteAction(
        obs_size, n_actions,
        n_hidden_channels=args.n_hidden_channels,
        n_hidden_layers=args.n_hidden_layers)
    # serializers.load_npz('dqn.npz', q_func)

    dev_buzzes = get_buzzes(q_func, dev_iter)
    dev_buzzes_dir = 'output/buzzer/rl/dev_buzzes.pkl'
    with open(dev_buzzes_dir, 'wb') as f:
        pickle.dump(dev_buzzes, f)
    print('Dev buzz {} saved to {}'.format(len(dev_buzzes), dev_buzzes_dir))

    report(dev_buzzes_dir)

if __name__ == '__main__':
    test()
