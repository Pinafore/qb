import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from qanta.util import constants as c
from qanta.buzzer.util import load_quizbowl
from qanta.buzzer.iterator import QuestionIterator
from qanta.buzzer.rnn_0 import dense_vector

class BuzzingGame(object):
    '''Environment for the buzzer.
    Each episode is initialize with a question.
    The observations are the guesses.
    The actions are waiting & buzzing.
    Args:
        iterator (QuestionIterator): iterator with batch size 1.
    '''

    def __init__(self, iterator):
        self.iterator = iterator
        self.observation_size = self.iterator.n_input
        self.action_space = spaces.Discrete(2)

    def reset(self, xp=np):
        '''Reset by sampling a new game from the iterator.
        Args:
            xp (numpy or cupy)
        Return:
            (ndarray): the first observation.
        '''
        self.pos = 0
        batch = self.iterator.next_batch(xp)
        self.vecs = batch.vecs.squeeze(1)
        self.results = batch.results.squeeze(1)
        self.last_position = self.results.shape[0] - 1
        self.position = 0
        return self.vecs[self.position]

    def finalize(self, reset=False):
        '''Finalize the iterator.'''
        self.iterator.finalize(reset=reset)

    def step(self, action):
        '''Forward one step in the game.
        Args:
            action: wait 0 or buzz 1.
        Return:
            observation (ndarray)
            reward (float)
            done (bool): terminal.
            info (dict): None.
        '''
        assert self.action_space.contains(action)
        if action == 0:
            reward = 0
        else:
            if self.results[self.position][0] == 1:
                reward = 2
            else:
                reward = -1

        self.position += 1
        done = (action == 1) or (self.position == self.last_position)
        return self.vecs[self.position], reward, done, None
        
def main():
    option2id, all_guesses = load_quizbowl()
    train_iter = QuestionIterator(all_guesses[c.BUZZER_DEV_FOLD], option2id,
            batch_size=1, make_vector=dense_vector)
    env = BuzzingGame(train_iter)
    env.reset()

if __name__ == '__main__':
    main()
