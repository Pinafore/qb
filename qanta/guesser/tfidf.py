from typing import List, Optional, Dict, Tuple
import os
from collections import defaultdict
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from qanta.guesser.abstract import AbstractGuesser
from qanta.datasets.abstract import QuestionText


class TfidfGuesser(AbstractGuesser):
    def __init__(self, config_num: Optional[int]):
        super().__init__(config_num)
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.i_to_ans = None

    def train(self, training_data) -> None:
        questions = training_data[0]
        answers = training_data[1]
        answer_docs = defaultdict(str)
        for q, ans in zip(questions, answers):
            text = " ".join(q)
            answer_docs[ans] += " " + text

        x_array = []
        y_array = []
        for ans, doc in answer_docs.items():
            x_array.append(doc)
            y_array.append(ans)

        self.i_to_ans = {i: ans for i, ans in enumerate(y_array)}
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3), min_df=2, max_df=0.9
        ).fit(x_array)
        self.tfidf_matrix = self.tfidf_vectorizer.transform(x_array)

    def guess(
        self, questions: List[QuestionText], max_n_guesses: Optional[int]
    ) -> List[List[Tuple[str, float]]]:
        representations = self.tfidf_vectorizer.transform(questions)
        guess_matrix = self.tfidf_matrix.dot(representations.T).T
        guess_scores = guess_matrix.max(axis=1).toarray().reshape(-1)
        guess_indices = np.array(guess_matrix.argmax(axis=1)).reshape(-1)
        guesses = []
        for i in range(len(questions)):
            idx = guess_indices[i]
            score = guess_scores[i]
            guesses.append([(self.i_to_ans[idx], score)])

        return guesses

    def save(self, directory: str) -> None:
        with open(os.path.join(directory, "params.pickle"), "wb") as f:
            pickle.dump(
                {
                    "config_num": self.config_num,
                    "i_to_ans": self.i_to_ans,
                    "tfidf_vectorizer": self.tfidf_vectorizer,
                    "tfidf_matrix": self.tfidf_matrix,
                },
                f,
            )

    @classmethod
    def load(cls, directory: str):
        with open(os.path.join(directory, "params.pickle"), "rb") as f:
            params = pickle.load(f)
            guesser = TfidfGuesser(params["config_num"])
            guesser.tfidf_vectorizer = params["tfidf_vectorizer"]
            guesser.tfidf_matrix = params["tfidf_matrix"]
            guesser.i_to_ans = params["i_to_ans"]
            return guesser

    @classmethod
    def targets(cls) -> List[str]:
        return ["params.pickle"]
