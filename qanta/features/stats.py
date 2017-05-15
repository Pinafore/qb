import numpy as np
import pickle

from qanta.features.abstract import AbstractFeatureExtractor
from qanta.util.constants import SENTENCE_STATS
from qanta.util.environment import QB_QUESTION_DB
from qanta.util.io import safe_open
from qanta.datasets.quiz_bowl import QuizBowlDataset, QuestionDatabase
import warnings


warnings.warn('old features extractors are deprecated and need to be rewritten', DeprecationWarning)


class StatsExtractor(AbstractFeatureExtractor):
    def __init__(self):
        super(StatsExtractor, self).__init__()
        with open(SENTENCE_STATS, 'rb') as f:
            self.word_count_mean, self.word_count_std = pickle.load(f)

        self.guess_frequencies = {}
        question_db = QuestionDatabase(QB_QUESTION_DB)
        all_questions = question_db.questions_with_pages()
        for page in all_questions:
            self.guess_frequencies[page] = sum(1 for x in all_questions[page] if x.fold == "train")

        self.frequency_mean = np.mean(list(self.guess_frequencies.values()))
        self.frequency_std = np.std(list(self.guess_frequencies.values()))
        for page in all_questions:
            normalized_frequency = normalize(
                self.guess_frequencies[page],
                self.frequency_mean,
                self.frequency_std
            )
            self.guess_frequencies[page] = normalized_frequency
        self.normed_missing_guess = normalize(0, self.frequency_mean, self.frequency_std)

    @property
    def name(self):
        return 'stats'

    def score_guesses(self, guesses, text):
        n_words = len(text.split())
        normalized_word_count = normalize(n_words, self.word_count_mean, self.word_count_std)
        for guess in guesses:
            formatted_guess = guess.replace(':', '').replace('|', '')
            normalized_guess_frequency = self.guess_frequencies.get(
                formatted_guess, self.normed_missing_guess)
            feature = '|stats guess_frequency:{} words_seen:{} norm_words_seen:{}'.format(
                normalized_guess_frequency, n_words, normalized_word_count)
            yield feature


def normalize(value, mean, var):
    return (value - mean) / var


def compute_question_stats(question_db_path: str):
    dataset = QuizBowlDataset(5, qb_question_db=question_db_path)
    train_dev_questions = dataset.questions_in_folds(('train', 'dev'))
    question_lengths = [len(q.flatten_text().split())
                        for q in train_dev_questions]

    mean = np.mean(question_lengths)
    std = np.std(question_lengths)

    stats = (mean, std)

    with safe_open(SENTENCE_STATS, 'wb') as f:
        pickle.dump(stats, f)
