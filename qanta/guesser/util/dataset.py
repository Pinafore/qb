from collections import defaultdict
from itertools import chain
import nltk
import os
import pickle

from qanta.guesser.util.preprocessing import preprocess_answer, preprocess_text
from qanta.util import constants
from qanta.util.environment import QB_QUESTION_DB
from qanta.util.io import pickle_cache
from qanta.util.qdb import QuestionDatabase


class Dataset:
    QUIZ_BOWL = 'QUIZ_BOWL'
    WIKI = 'WIKI'


def _fname_maker(datasets):
    return os.path.join(constants.ID_MAP_DIR, '+'.join(sorted(datasets)))


@pickle_cache(_fname_maker)
def get_or_make_id_map(datasets):
    """Creates a map for converting text to ids. The result is cached to a file to speed up future calls.
    Args:
        datasets: An iterable of dataset keys. Ex: [qanta.guesser.Dataset.QUIZ_BOWL]
    Returns:
        A dict mapping words to IDs. The data used comes from qanta.util.question_iterator.
    """
    word_to_id = defaultdict()
    word_to_id.default_factory = word_to_id.__len__
    data = get_all_questions(datasets, folds=['train'])

    # Iterate over all words in every question + answer. Multi-word answers will still be just one "word".
    for dataset in data.values():
        for question, answer in dataset:
            word_to_id[answer]
            for sent in question.values():
                for w in sent.split():
                    word_to_id[w]

    return dict(word_to_id)


def sentences_from_page(wiki_page):
    for sent in nltk.sent_tokenize(wiki_page.content):
        if sent == "== See also ==":
            break
        elif not sent or sent.startswith("=="):
            continue
        yield sent


def _qb_generator(db, pages, fold):
    for i, q in db.query('from questions where page != "" and fold == ?', (fold,), text=True).items():
        if q.page not in pages:
            continue

        yield {i: preprocess_text(text) for i, text in q.text.items()}, preprocess_answer(q.page)


def _wiki_generator(pages):
    with open(constants.DOMAIN_OUTPUT, 'rb') as f:
        for sentence, page in pickle.load(f):
            yield {0: preprocess_text(sentence)}, preprocess_answer(page)


def get_all_questions(datasets, folds=None):
    """Returns a dict mapping dataset name to a generator which yields questions, answer tuples. All question runs are included."""
    if folds is None:
        folds = ['train']
    db = QuestionDatabase(QB_QUESTION_DB)
    pages = set(db.page_by_count(min_count=constants.MIN_APPEARANCES))

    result = {}
    if Dataset.QUIZ_BOWL in datasets:
        result[Dataset.QUIZ_BOWL] = chain.from_iterable(_qb_generator(db, pages, fold=fold) for fold in folds)

    if Dataset.WIKI in datasets:
        result[Dataset.WIKI] = _wiki_generator(pages)

    return result
