import typing
from collections import namedtuple
from abc import ABCMeta, abstractmethod
from qanta.datasets.quiz_bowl import Question as TossUpQuestion
from qanta.new_expo.agent import HumanAgent, GuesserBuzzerAgent
from qanta.new_expo.hook import GameInterfaceHook, EndOfRoundHook,\
                                NotifyBuzzingHook


class Game(object):
    '''A Game object represents a complete QuizBowl game with multiple rounds, each
    round is a Round object.
    Components of game state:
        a Round object which contains the question and the progress (position)
        history of actions of the agents
        scores of the agents
    Args:
        quetion_iter: 
        agents: players
        hooks: visualization stuff
    Process:
        At the beginning of each round, the Game object creats a Round object
        using a new question from question iterator. At each step, the Game
        object queries the Round object for an updated clue. The clue will be
        given in the form of partial question which depends on the type of the
        question.
        Game control sends the updated clue to all agents, agents update their
        actions, and game control retrieves the actions. 
        Each action has two parts, [buzz, guess].
        Break tie.
        If any action is not None, game control sends the answer to Round to
        evaluate. Round returns the correctness, game control determines
        termination and rewards.
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
        self.hooks.append(GameInterfaceHook(self))
        self.hooks.append(EndOfRoundHook(self))
        self.hooks.append(NotifyBuzzingHook(self))
        self.round_hooks = [x for x in self.hooks if x.call_every == 'round']
        self.step_hooks = [x for x in self.hooks if x.call_every == 'step']

    def evaluate(self, agent):
        guess = agent.action.guess
        if guess is None:
            guess = ''
        if isinstance(agent, HumanAgent):
            # guess is used to indicate player number for human agent
            response = input("Player {}, provide an answer:\t".format(guess))
            # FIXME only accept + and -
            if '+' in response:
                result = True
            else:
                result = False
        else:
            print('QANTA: {}'.format(guess))
            result = self.round.evaluate(guess)
        return result
    
    def run_hooks(self, call_every):
        for hook in self.hooks:
            if hook.call_every == call_every:
                hook.run()

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

            if state is None:
                terminate = True
                # end of question
                # find an agent that has not buzzed, check guess
                for i, x in enumerate(self.agents):
                    if self.buzzed[i] is False:
                        if self.evaluate(x):
                            self.scores[i] += 10
                        break
            
            if terminate:
                break
            
            for i, x in enumerate(self.agents):
                x.update(state)
                if self.buzzed[i] is False and x.action.buzz is True:
                    self.buzzed[i] = True
                    if self.evaluate(x):
                        self.scores[i] += 10
                        terminate = True
                        break
                    else:
                        self.scores[i] -= 5

            print(self.buzzed)
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
            return ' '.join(self.question_text[:self.position])

class BonusRound(Round):
    pass
