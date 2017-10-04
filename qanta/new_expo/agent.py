import numpy as np
from collections import namedtuple
import typing
from typing import List, Dict, Tuple, Optional
from abc import ABCMeta, abstractmethod
from qanta.datasets.quiz_bowl import Question as TossUpQuestion
from qanta.new_expo.util import interpret_keypress

import chainer
import chainer.functions as F
from qanta.buzzer.iterator import QuestionIterator
from qanta.buzzer.util import load_quizbowl, GUESSERS
from qanta.buzzer.models import MLP, RNN
N_GUESSERS = len(GUESSERS)
N_GUESSES = 10

Action = namedtuple('Action', ['buzz', 'guess'])


class ThresholdBuzzer:
    
    def __init__(self, threshold=1.2):
        self.threshold = threshold

    def new_round(self):
        pass

    def buzz(self, guesses):
        '''guesses is a sorted list of (guess, score)'''
        return guesses[0][1] > self.threshold


class ESGuesserWrapper:

    def __init__(self, guesser):
        self.guesser = guesser
        self.guesses = None

    def new_round(self):
        pass

    def guess(self, text):
        guesses = self.guesser.guess_single(text)
        self.guesses = sorted(guesses.items(), key=lambda x: x[1])[::-1]
        return guesses

def dense_vector(scores: Dict[str, float], prev_scores=None):
    N_GUESSES = 10

    if prev_scores is None:
        prev_scores = dict()
    prev_vec = sorted(prev_scores.items(), key=lambda x: x[1])[::-1]
    prev_vec = [x[1] for x in prev_vec]
    _len = N_GUESSES - len(prev_vec)
    if _len > 0:
         prev_vec += [0 for _ in range(_len)]
            
    vec = []
    diff_vec = []
    isnew_vec = []
    scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
    for guess, score in scores:
        vec.append(score)
        if guess in prev_scores:
            diff_vec.append(score - prev_scores[guess])
            isnew_vec.append(0)
        else:
            diff_vec.append(score) 
            isnew_vec.append(1)
    if len(scores) < N_GUESSES:
        for k in range(max(N_GUESSES - len(scores), 0)):
            vec.append(0)
            diff_vec.append(0)
            isnew_vec.append(0)
    features = [
            vec[0], vec[1], vec[2],
            isnew_vec[0], isnew_vec[1], isnew_vec[2],
            diff_vec[0], diff_vec[1], diff_vec[2],
            vec[0] - vec[1], vec[1] - vec[2], 
            vec[0] - prev_vec[0],
            sum(isnew_vec[:5]),
            np.average(vec), np.average(prev_vec),
            np.average(vec[:5]), np.average(prev_vec[:5]),
            np.var(vec), np.var(prev_vec),
            np.var(vec[:5]), np.var(prev_vec[:5])
            ]

    return features


class RNNBuzzer:

    def __init__(self, word_skip=0):
        self.word_skip = word_skip
        model_dir = 'output/buzzer/neo/neo_0.npz'
        model = RNN(21, 300, 2)
        print('QANTA: loading model')
        chainer.serializers.load_npz(model_dir, model)

        chainer.cuda.get_device(0).use()
        model.to_gpu(0)

        self.model = model
        self.new_round()

    def new_round(self):
        self.model.reset_state()
        self.prev_buzz = None
        self.prev_scores = None
        self.ys = None
        self.skipped = 0

    def buzz(self, guesses: Dict[str, float]):
        if self.skipped < self.word_skip:
            self.skipped += 1
            return self.prev_buzz
        self.skipped = 0
        feature_vec = dense_vector(guesses, self.prev_scores)
        self.prev_scores = guesses
        # batch_size=1, dim
        vec = self.model.xp.asarray([feature_vec], dtype=self.model.xp.float32) 
        ys = self.model.step(vec) # batch_size=1, 2
        ys = F.softmax(ys)
        ys.to_cpu()
        ys = ys[0].data
        self.ys = ys
        buzz = ys[0] > ys[1]
        self.prev_buzz = buzz
        return buzz


class Agent:
    __metaclass__ = ABCMeta

    @abstractmethod
    def new_round(self):
        '''Initialize for a new question'''
        pass
    
    @abstractmethod
    def update(self, state):
        '''Update the agent state and action based on the state'''
        pass

    @abstractmethod
    def new_round(self):
        '''Reset agent for a new round'''
        pass

    @abstractmethod
    def notify_buzzing(self, buzzed):
        '''Notify agent when opponent buzzes with a list
        the first entry corresponds the agent itself'''
        pass
        

class GuesserBuzzerAgent(Agent):

    def __init__(self, guesser, buzzer):
        self.guesser = guesser
        self.buzzer = buzzer
        self.action = Action(False, None)
        self.all_guesses = [] # internal state used for things like visualization
        print("I'm ready too")
        self.n_steps = 0
        self.opponent_buzzed = False

    def notify_buzzing(self, buzzed):
        self.opponent_buzzed = all(buzzed[1:])

    def new_round(self):
        self.action = Action(False, None)
        self.all_guesses = []
        self.guesser.new_round()
        self.buzzer.new_round()
        self.n_steps = 0

    def update(self, state):
        guesses = self.guesser.guess(state)
        if isinstance(guesses, list):
            guesses = {x[0]:x[1] for x in guesses}
        self.all_guesses.append(guesses)
        buzz = self.buzzer.buzz(guesses)
        # buzz = (self.n_steps > 70) and\
        # don't buzz if opponent has
        buzz = buzz and (not self.opponent_buzzed)
        guesses = sorted(guesses.items(), key=lambda x: x[1])[::-1]
        self.all_guesses.append(guesses)
        self.action = Action(buzz, guesses[0][0])
        self.n_steps += 1

class HumanAgent(Agent):
    '''Human agent gets evaluated based on keypress
    guess in an action of human agent indicates the player number'''

    def __init__(self):
        self.action = Action(False, None)
        self._initial_key_test()

    def _initial_key_test(self):
        players_needed = [1, 2, 3, 4]
        current_players = set()
        while len(current_players) < len(players_needed):
            print("Player %i, please buzz in" % min(x for x in players_needed if x not in current_players))
            press = interpret_keypress()
            if press in players_needed:
                # os.system("afplay /System/Library/Sounds/Glass.aiff")
                print("Thanks for buzzing in, player %i!" % press)
                current_players.add(press)

    def new_round(self):
        self.action = Action(False, None)

    def update(self, state):
        press = interpret_keypress()
        if isinstance(press, int):
            self.action = Action(True, press)
        else:
            self.action = Action(False, None)
