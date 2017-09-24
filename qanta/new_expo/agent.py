import typing
from collections import namedtuple
from abc import ABCMeta, abstractmethod
from qanta.datasets.quiz_bowl import Question as TossUpQuestion
from qanta.new_expo.util import interpret_keypress

Action = namedtuple('Action', ['buzz', 'guess'])

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
        self.me_buzzed = False

    def notify_buzzing(self, buzzed):
        self.me_buzzed = buzzed[0]
        self.opponent_buzzed = all(buzzed[1:])

    def new_round(self):
        self.action = Action(False, None)
        self.all_guesses = []
        self.guesser.new_round()
        self.buzzer.new_round()
        self.n_steps = 0

    def update(self, state):
        guesses = self.guesser.guess(state)
        if isinstance(guesses, dict):
            guesses = list(sorted(guesses.items(), key=lambda x: x[1]))
            guesses = guesses[::-1]
        self.all_guesses.append(guesses)
        # TODO
        # buzz = self.buzzer.buzz(guesses)
        buzz = (self.n_steps > 70) and\
                (not self.me_buzzed) and\
                (not self.opponent_buzzed)
        self.action = Action(buzz, guesses[0][0])
        self.n_steps += 1

class HumanAgent(Agent):

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
