import typing
from collections import namedtuple
from abc import ABCMeta, abstractmethod
from qanta.datasets.quiz_bowl import Question as TossUpQuestion
from qanta.expo.buzzer import interpret_keypress
from qanta.new_expo.util import GetchUnix

Action = namedtuple('buzz', 'guess')

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

class GuesserBuzzerAgent(Agent):

    def __init__(self, guesser, buzzer):
        self.guesser = guesser
        self.buzzer = buzzer
        self.action = Action(False, None)
        self.all_guesses = [] # internal state used for things like visualization

    def new_round(self):
        self.action = Action(False, None)

    def update(self, state):
        guesses = self.guesser.guess(state)
        if isinstance(guesses, dict):
            guesses = list(sorted(guesses.items(), key=lambda x: x[1]))
        self.all_guesses.append(guesses)
        # TODO
        buzz = False
        self.action = Action(buzz, guesses[0][0])

class HumanAgent(Agent):

    def __init__(self):
        self.getch = _GetchUnix()
        self.action = Action(False, None)
        self._initial_key_test()

    def _initial_key_test(self):
        players_needed = [1, 2, 3, 4]
        while len(current_players) < len(players_needed):
            print("Player %i, please buzz in" % min(x for x in players_needed if x not in current_players))
            press = interpret_keypress()
            if press in players_needed:
                os.system("afplay /System/Library/Sounds/Glass.aiff")
                print("Thanks for buzzing in, player %i!" % press)
                current_players.add(press)

    def interpret_keypress():
        """
        See whether a number was pressed (give terminal bell if so) and return
        value.  Otherwise returns none.  Tries to handle arrows as a single
        press.
        """
        press = self.getch()
    
        if press == 'Q':
            raise Exception('Exiting expo by user request from pressing Q')
    
        if press == '\x1b':
            getch()
            getch()
            press = "direction"
    
        if press != "direction" and press != " ":
            try:
                press = int(press)
            except ValueError:
                press = None
        return press
    
    def new_round(self):
        self.action = Action(False, None)

    def update(self):
        press = interpret_keypress()
        if isinstance(press, int):
            response = input("Player %i, provide an answer:\t" % press)
        self.action = Action(True, None)
