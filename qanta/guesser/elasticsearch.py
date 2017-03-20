from typing import List, Optional
import progressbar
from qanta.datasets.abstract import QuestionText
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.guesser.abstract import AbstractGuesser
from qanta.preprocess import format_guess
from qanta.search.elasticsearch import ElasticSearchIndex


class ElasticSearchGuesser(AbstractGuesser):
    def qb_dataset(self):
        return QuizBowlDataset(1)

    def train(self, training_data):
        documents = {}
        for sentences, ans in zip(training_data[0], training_data[1]):
            page = format_guess(ans)
            paragraph = ' '.join(sentences)
            if page in documents:
                documents[page] += ' ' + paragraph
            else:
                documents[page] = paragraph
        ElasticSearchIndex.build(documents)

    def guess(self,
              questions: List[QuestionText],
              max_n_guesses: Optional[int]):
        predictions = []
        es_index = ElasticSearchIndex()
        bar = progressbar.ProgressBar()
        for q in bar(questions):
            guesses = es_index.search(q)[:max_n_guesses]
            predictions.append(guesses)
        return predictions

    @classmethod
    def targets(cls):
        return []

    @classmethod
    def load(cls, directory: str):
        return ElasticSearchGuesser()

    def save(self, directory: str):
        pass
