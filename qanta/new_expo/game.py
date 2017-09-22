import typing
from collections import namedtuple
from abc import ABCMeta, abstractmethod
from qanta.datasets.quiz_bowl import Question as TossUpQuestion
from qanta.new_expo.agent import HumanAgent, GuesserBuzzerAgent

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
    def __init__(self, question_list, agents, hooks):
        self.question_iter = iter(question_list)
        self.agents = agents
        self.hooks = self.hooks
        self.scores = [0 for _ in agents]

    def evaluate(self, agent, guess):
        if isinstance(agent, HumanAgent):
            response = input("Player, provide an answer:\t")
            if '+' in response:
                result = True
            elif '-' in response:
                result = False
        elif guess is None:
            result = False
        else:
            result = self.round.evaluate(guess)
        return result

    def run_round(self):
        question = next(self.question_iter)
        if isinstance(quetion, TossUpQuestion):
            self.round = TossUpRound(question)
        else:
            raise ValueError("BonusQuestion is not yet supported")

        for x in self.agents:
            x.new_round()
        buzzed = [False for _ in self.agents]

        while True:
            state = self.round.next()

            # end of question
            if state is None:
                # find an agent that has not buzzed, check guess
                for i, x in enumerate(self.agents):
                    if buzzed[i] is False:
                        if self.evaluate(x.action.guess):
                            self.scores[i] += 10
                        break
                return
            
            terminate = False
            for i, x in enumerate(self.agents):
                x.update(state)
                if x.action.buzz is True:
                    buzzed[i] = True
                    if self.evaluate(x.action.guess):
                        self.scores[i] += 10
                        terminate = True
                        break
                    else:
                        self.scores[i] -= 5
            if terminate:
                return

    def run(self, n_rounds):
        for round_num in range(n_rounds):
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
        self.length = len(question_text)

    def evaluate(self, guess: str):
        if guess == question.answer or guess == question.page:
            return True
        else:
            return False

    def __next__(self):
        self.position += 1
        if self.position > self.length:
            return None
        else:
            return ' '.join(self.question_text[:self.position])

class BonusRound(Round):
    pass
