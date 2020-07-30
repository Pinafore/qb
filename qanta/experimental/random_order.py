import random
import pickle

from qanta.config import conf
from qanta.util.io import safe_path
from qanta.util.multiprocess import _multiprocess
from qanta.guesser.abstract import AbstractGuesser
from qanta.datasets.quiz_bowl import QuizBowlDataset, Question
from qanta.guesser.experimental.elasticsearch_instance_of import (
    ElasticSearchWikidataGuesser,
)

"""Randomly shuffle the word order and see if it changes the guesses.
"""

gspec = AbstractGuesser.list_enabled_guessers()[0]
guesser_dir = AbstractGuesser.output_path(gspec.guesser_module, gspec.guesser_class, "")
guesser = ElasticSearchWikidataGuesser.load(guesser_dir)


def main():
    fold = "guessdev"
    db = QuizBowlDataset(1, guesser_train=True, buzzer_train=True)
    questions = db.questions_in_folds([fold])
    first_n = lambda x: len(x)

    print(guesser.guess_single(" ".join(questions[0].text.values())))

    """
    s = [0, 0, 0, 0, 0]
    for q in questions:
        sents = list(q.text.values())
        text_before = ' '.join(sents[:first_n(sents)])
        words = text.split()
        random.shuffle(words)
        text_after = ' '.join(words)
        gb = guesser.guess_single(text_before)
        ga = guesser.guess_single(text_after)
        
        s[0] += gb[0][0] == q.page # accuracy before
        s[1] += ga[0][0] == q.page # accuracy after
        s[2] += ga[0][0] == gb[0][0] # if generate same guess
        s[3] += q.page in [x[0] for x in gb[:5]] # top 5 accuracy before
        s[4] += q.page in [x[0] for x in ga[:5]] # top 5 accuracy after
    """


if __name__ == "__main__":
    main()
