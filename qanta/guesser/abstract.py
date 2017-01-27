import os
from collections import defaultdict
from abc import ABCMeta, abstractmethod
from typing import List, Dict, Tuple

import pandas as pd

from qanta.datasets.abstract import TrainingData, QuestionText, Answer, AbstractDataset
from qanta.datasets.quiz_bowl import QuizBowlEvaluationDataset
from qanta.util import constants as c


class AbstractGuesser(metaclass=ABCMeta):
    def __init__(self):
        """
        Abstract class representing a guesser. All abstract methods must be implemented. Class
        construction should be light and not load data since this is reserved for the
        AbstractGuesser.load method.

        self.parallel tells qanta whether or not this guesser should be parallelized.

        """
        self.parallel = False

    @property
    @abstractmethod
    def requested_datasets(self) -> Dict[str, AbstractDataset]:
        """
        Return a mapping of requested datasets. For each entry in the dictionary
        AbstractDataset.training_data will be called and its result stored using the str key for use
        in AbstractGuesser.train
        :return:
        """
        pass

    @abstractmethod
    def train(self, training_data: Dict[str, TrainingData]) -> None:
        """
        Given training data, train this guesser so that it can produce guesses.

        The training_data dictionary is keyed by constants from Datasets such as Datasets.QUIZ_BOWL.
        The provided data to this method is based on the requested list of datasets from
        self.requested_datasets.

        The values of these keys is a tuple of two elements which can be seen as (train_x, train_y).
        In this case train_x is a list of question runs. For example, if the answer for a question
        is "Albert Einstein" the runs might be ["This", "This German", "This German physicist", ...]
        train_y is a list of true labels. The questions are strings and the true labels are strings.
        Labels are in canonical form. Questions are not preprocessed in any way. To implement common
        pre-processing refer to the qanta/guesser/preprocessing module.

        :param training_data: training data in the format described above
        :return: This function does not return anything
        """
        pass

    @abstractmethod
    def guess(self,
              questions: List[QuestionText],
              max_n_guesses: int) -> List[List[Tuple[Answer, float]]]:
        """
        Given a list of questions as text, return n_guesses number of guesses per question. Guesses
        must be returned in canonical form, are returned with a score in which higher is better, and
        must also be returned in sorted order with the best guess (highest score) at the front of
        the list and worst guesses (lowest score) at the bottom.

        It is guaranteed that before AbstractGuesser.guess is called that either
        AbstractGuesser.train is called or AbstractGuesser.load is called.

        :param questions: Questions to guess on
        :param max_n_guesses: Number of guesses to produce per question
        :return: List of top guesses per question
        """
        pass

    @classmethod
    @abstractmethod
    def targets(cls) -> List[str]:
        """
        List of files located in directory that are produced by the train method and loaded by the
        save method.
        :return: list of written files
        """
        pass

    @classmethod
    def files(cls, directory: str) -> List[str]:
        return [os.path.join(directory, file) for file in cls.targets()]

    @classmethod
    @abstractmethod
    def load(cls, directory: str):
        """
        Given the directory used for saving this guesser, create a new instance of the guesser, and
        load it for guessing or scoring.

        :param directory: training data for guesser
        :return: Instance of AbstractGuesser ready for calling guess/score
        """
        pass

    @abstractmethod
    def save(self, directory: str) -> None:
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """
        Return the display name of this guesser which is used in reporting scripts to identify this
        particular guesser
        :return: display name of this guesser
        """
        pass

    def generate_guesses(self, max_n_guesses: int, folds: List[str]) -> pd.DataFrame:
        """
        Generates guesses for this guesser for all questions in specified folds and returns it as a
        DataFrame

        WARNING: this method assumes that the guesser has been loaded with load or trained with
        train. Unexpected behavior may occur if that is not the case.
        :param max_n_guesses: generate at most this many guesses per question, sentence, and token
        :param folds: which folds to generate guesses for
        :return: dataframe of guesses
        """
        dataset = QuizBowlEvaluationDataset()
        questions_by_fold = dataset.questions_by_fold()

        q_folds = []
        q_qnums = []
        q_sentences = []
        q_tokens = []
        question_texts = []

        for fold in folds:
            questions = questions_by_fold[fold]
            for q in questions:
                for sent, token, text_list in q.partials():
                    text = ' '.join(text_list)
                    question_texts.append(text)
                    q_folds.append(fold)
                    q_qnums.append(q.qnum)
                    q_sentences.append(sent)
                    q_tokens.append(token)

        guesses_per_question = self.guess(question_texts, max_n_guesses)

        if len(guesses_per_question) != len(question_texts):
            raise ValueError('Guesser not returning the right number of answers')

        df_qnums = []
        df_sentences = []
        df_tokens = []
        df_guesses = []
        df_scores = []
        df_folds = []
        df_guessers = []
        guesser_name = self.display_name

        for i in range(len(question_texts)):
            guesses_with_scores = guesses_per_question[i]
            fold = q_folds[i]
            qnum = q_qnums[i]
            sentence = q_sentences[i]
            token = q_tokens[i]
            for guess, score in guesses_with_scores:
                df_qnums.append(qnum)
                df_sentences.append(sentence)
                df_tokens.append(token)
                df_guesses.append(guess)
                df_scores.append(score)
                df_folds.append(fold)
                df_guessers.append(guesser_name)

        return pd.DataFrame({
            'qnum': df_qnums,
            'sentence': df_sentences,
            'token': df_tokens,
            'guess': df_guesses,
            'score': df_scores,
            'fold': df_folds,
            'guesser': df_guessers
        })

    @staticmethod
    def guess_path(directory: str, fold: str) -> str:
        return os.path.join(directory, 'guesses_{}.pickle'.format(fold))

    @staticmethod
    def save_guesses(guess_df: pd.DataFrame, directory: str):
        folds = ['train', 'dev', 'test', 'devtest']
        for fold in folds:
            fold_df = guess_df[guess_df.fold == fold]
            output_path = AbstractGuesser.guess_path(directory, fold)
            fold_df.to_pickle(output_path)

    @staticmethod
    def load_guesses(directory: str, folds=c.ALL_FOLDS) -> pd.DataFrame:
        """
        Loads all the guesses pertaining to a guesser inferred from directory
        :param directory: where to load guesses from
        :param folds: folds to load, by default all of them
        :return: guesses across all folds for given directory
        """
        assert len(folds) > 0
        guess_df = None
        for fold in folds:
            input_path = AbstractGuesser.guess_path(directory, fold)
            if guess_df is None:
                guess_df = pd.read_pickle(input_path)
            else:
                new_guesses_df = pd.read_pickle(input_path)
                guess_df = pd.concat([guess_df, new_guesses_df])

        return guess_df

    @staticmethod
    def load_all_guesses(directory_prefix='') -> pd.DataFrame:
        """
        Loads all guesses from all guessers and folds
        :return:
        """
        guess_df = None
        for guesser_class, _ in c.GUESSER_LIST:
            input_path = os.path.join(directory_prefix, c.GUESSER_TARGET_PREFIX, guesser_class)
            if guess_df is None:
                guess_df = AbstractGuesser.load_guesses(input_path)
            else:
                new_guess_df = AbstractGuesser.load_guesses(input_path)
                guess_df = pd.concat([guess_df, new_guess_df])

        return guess_df

    @staticmethod
    def load_guess_score_map(guess_df: pd.DataFrame) -> defaultdict:
        guess_score_map = defaultdict(dict)
        for row in guess_df.itertuples():
            guess_score_map[row.guesser][(row.qnum, row.sentence, row.token, row.guess)] = row.score

        return guess_score_map
