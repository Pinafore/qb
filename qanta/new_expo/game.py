import typing
from collections import namedtuple
from abc import ABCMeta, abstractmethod
from qanta.datasets.quiz_bowl import Question as TossUpQuestion
from qanta.new_expo.agent import HumanAgent, GuesserBuzzerAgent
from qanta.new_expo.hook import GameInterfaceHook, NotifyBuzzingHook


class Game(object):
    '''A Game object represents a QuizBowl game with multiple rounds, where
    each round is a Round object.
    Game process:
    - Beginning of a round
        - Game control creats a Round object with a new question
        - Game control notifies all players the start of a new round
    - In the middle of a Round, at each step,
        - Game control queries the Round object for an updated clue
        - Game sends the updated clue to all agents
        - Agents update their actions. Each action is [buzz, guess]
        - Game control retrieves the actions
        - If any agent chooses to buzz, game control sends the agent to Round
          to get an evaluation. 
        - Round returns True if the guess is correct, and False otherwise
        - Game control determines termination and rewards
        - Run all per-step hooks
    - End of a round
        - Game control update its states
        - Run all the per-round hooks
    Args:
        quetion_list: a list of TossUpQuestions and BonusQuestions
        agents: players
        hooks: hooks that are callable are ones that need to created by passing
        the game control itself, they are internal hooks 
    '''
    def __init__(self, question_list, agents, hooks=[]):
        self.question_iter = iter(question_list)
        self.agents = agents
        self.hooks = hooks
        self.scores = [0 for _ in agents]
        self.buzz = [False for _ in agents]
        self.round = None
        self.round_num = 0
        self.setup_hooks()

    def setup_hooks(self):
        '''Setup some extra hooks that need access to the internal state of
        game control'''
        # create internal hooks that requires access to game control 
        for i in range(len(self.hooks)):
            if callable(self.hooks[i]):
                self.hooks[i] = self.hooks[i](self)

        hooks = {'round': [x for x in self.hooks if x.call_every == 'round'], 
                 'step': [x for x in self.hooks if x.call_every == 'step']}
        self.hooks = hooks

    def evaluate(self, agent):
        '''Evaluate an agent's guess'''
        guess = agent.action.guess
        if guess is None:
            guess = ''
        if isinstance(agent, HumanAgent):
            '''In the case of human agent, guess is an integer that indicates
            the player number in the human team. The evaluation of human agent
            is determined by keyboard input.'''
            print()
            print('=============================')
            response = input("Player {}, provide an answer:\t".format(guess))
            # FIXME only accept + and -
            if '+' in response:
                result = True
            else:
                result = False
        else:
            print()
            print('=============================')
            print('QANTA: {}'.format(guess))
            print('=============================')
            print()
            result = self.round.evaluate(guess)
        return result
    
    def run_hooks(self, call_every):
        for hook in self.hooks[call_every]:
            hook.run()

    def break_tie(self, indices):
        if len(indices) > 0:
            return indices[0]
        else:
            return None

    def run_round(self):
        # get new question, set up the round
        question = next(self.question_iter)
        if isinstance(question, TossUpQuestion):
            self.round = TossUpRound(question)
        else:
            raise ValueError("BonusQuestion is not yet supported")

        # reset players
        for x in self.agents:
            x.new_round()
        self.buzzed = [False for _ in self.agents]

        # start round
        step_num = 0
        while True:
            step_num += 1
            state = self.round.next()
            terminate = False

            not_buzzed = [i for i, x in enumerate(self.buzzed) if not x]

            if state is None:
                terminate = True
                # end of question
                if len(not_buzzed) > 0:
                    i = self.break_tie(not_buzzed)
                    if self.evaluate(self.agents[i]):
                        self.scores[i] += 10
                break
            
            for i, x in enumerate(self.agents):
                x.update(state)

            buzzing = [i for i, x in enumerate(self.agents) if x.action.buzz]
            buzzing = [i for i in buzzing if i in not_buzzed]
            if len(buzzing) > 0:
                i = self.break_tie(buzzing)
                self.buzzed[i] = True
                if self.evaluate(self.agents[i]):
                    self.scores[i] += 10
                    terminate = True
                    break
                else:
                    self.scores[i] -= 5

            if all(self.buzzed) or terminate:
                break

            self.run_hooks('step')
        # end of round
        self.run_hooks('round')

    def run(self, n_rounds):
        for round_num in range(n_rounds):
            self.round_num = round_num + 1
            self.run_round()

class Round(object): 
    '''A Round object represents a single question in the game. 
    State of the round is determined by a Quetion object and the progress.
    A Round can be seen as a RL game environment.
    It's either a TossUpRound or a BonusRound.
    '''
    def __init__(self, question):
        self.question = question

class TossUpRound(Round):

    def __init__(self, question: TossUpQuestion):
        self.question = question
        self.position = 0
        self.action = None
        self.question_text = self.question.flatten_text().split()
        self.length = len(self.question_text)

    def evaluate(self, guess: str):
        if guess == self.question.answer or guess == self.question.page:
            return True
        else:
            return False

    def get_clue(self):
        return ' '.join(self.question_text[:self.position])

    def get_answer(self):
        return self.question.page

    def next(self):
        self.position += 1
        if self.position > self.length:
            return None
        else:
            return self.get_clue()

class BonusRound(Round):
    pass
