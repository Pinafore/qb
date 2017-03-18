from typing import List, Tuple, Optional
from itertools import repeat
from multiprocessing import Pool

from qanta.datasets.abstract import TrainingData, Answer, QuestionText
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.guesser.abstract import AbstractGuesser
from qanta.preprocess import format_guess
from qanta.whoosh_ir import WhooshWikiIndex
from qanta.util.constants import WHOOSH_WIKI_INDEX_PATH
import progressbar


_whoosh_reference = {}


def _par_search(text, limit):
    if 'ix' in _whoosh_reference:
        ix = _whoosh_reference['ix']
    else:
        ix = WhooshWikiIndex()
        _whoosh_reference[ix] = ix
    return ix.search(text, limit)


class WhooshWikiGuesser(AbstractGuesser):
    def qb_dataset(self):
        return QuizBowlDataset(1)

    def guess(self,
              questions: List[QuestionText],
              max_n_guesses: Optional[int]) -> List[List[Tuple[Answer, float]]]:
        if max_n_guesses is None:
            limit = 50
        else:
            limit = max_n_guesses

        bar = progressbar.ProgressBar()
        pool = Pool(processes=20)
        return list(pool.starmap(_par_search, zip(questions, repeat(limit)), chunksize=1000))

    def train(self, training_data: TrainingData) -> None:
        documents = {}
        for sentence, ans in zip(training_data[0], training_data[1]):
            page = format_guess(ans)
            paragraph = ' '.join(sentence)
            if page in documents:
                documents[page] += ' ' + paragraph
            else:
                documents[page] = paragraph
        WhooshWikiIndex.build(documents, index_path=WHOOSH_WIKI_INDEX_PATH)

    @classmethod
    def targets(cls) -> List[str]:
        return []

    @classmethod
    def raw_targets(cls):
        return [WHOOSH_WIKI_INDEX_PATH]

    @classmethod
    def load(cls, directory: str):
        """
        There is nothing to load since the index is saved on the file system
        """
        return WhooshWikiGuesser()

    def save(self, directory: str) -> None:
        """
        There is nothing to save since the index is saved on the file system
        """
        pass
