from enum import Enum
from typing import Optional, List
from qanta.datasets.quiz_bowl import Question as QBQuestion


class AnswerStatus(Enum):
    UNANSWERED = 1
    CORRECT = 2
    WRONG = 3

class Player:
    def __init__(self, player_id: int, name: str, score: int, answer_status: AnswerStatus, is_human: bool):
        self.id = player_id
        self.name = name
        self.score = score
        self.answer_status = answer_status
        self.is_human = is_human


class Buzz:
    def __init__(self, player_id: int, correct: bool, guess: str):
        self.player_id = player_id
        self.correct = correct
        self.guess = guess


class Question:
    def __init__(self, question: QBQuestion):
        self.question = question
        self.sentence = 0
        self.word = 0

    @property
    def is_end_of_question(self):
        return False


class GameState:
    def __init__(self, game_id: int, players: List[Player], buzzes: List[Buzz], question: Optional[Question]):
        self.game_id = game_id
        self.players = players
        self.buzzes = buzzes
        self.question = question


class Environment:
    def __init__(self, questions: List[QBQuestion], players: List[Player]):
        self.questions = questions
        self.game_state = GameState(0, players, [], None)

    def _validate(self, action_tuple):
        action, data = action_tuple
        if action == Actions.BUZZ:
            if not isinstance(data, str):
                raise ValueError('data must be a string representing a guess')
        elif action == Actions.WAIT:
            if data is not None:
                raise ValueError('When waiting data must be None')
        else:
            raise ValueError('Action must be BUZZ or WAIT')

    def _buzz(self, guess):
        pass

    def _wait(self):
        pass

    def _next_question(self):
        if self.question_index is None:
            self.question_index = 1
            qb_question = self.questions[self.question_index - 1]
            question = Question(self.question_index, qb_question.id, )


    def step(self, action_tuple):
        self._validate(action_tuple)
        action, data = action_tuple
        if action == Actions.BUZZ:
            guess = data
        elif action == Actions.WAIT:
            pass

class Actions(Enum):
    BUZZ = 1
    WAIT = 2
    NEXT_QUESTION = 3
