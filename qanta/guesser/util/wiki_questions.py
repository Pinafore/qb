from contextlib import ExitStack
import heapq
from itertools import chain, repeat
import pickle
import random

from qanta import logging
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.util.constants import DOMAIN_PREDICTIONS_PREFIX, DOMAIN_OUTPUT, DOMAIN_TARGET_PREFIX
from qanta.util.io import safe_open
from qanta.preprocess import clean_question
from qanta.wikipedia.cached_wikipedia import CachedWikipedia

import nltk

log = logging.get(__name__)


def sentences_from_page(wiki_page):
    for sent in nltk.sent_tokenize(wiki_page.content):
        if sent == "== See also ==":
            break
        elif not sent or sent.startswith("=="):
            continue
        yield sent


def generate_domain_classifier_data(weight=150):
    """
    Reads all sentences from every wikipedia page corresponding to a known answer and splits them into two vowpal wabbit files,

    interleaving true quiz bowl questions randomly and with higher weight specified by the weight arg.
    """
    qb_data = QuizBowlDataset(guesser_train=True).training_data()
    real_questions = [('1', str(weight), ans, clean_question(sent)) for q, ans, _ in zip(*qb_data) for sent in q]
    pages = set(a for _, _, a, _ in real_questions)

    cw = CachedWikipedia()

    # Split wikipedia questions into two sets
    wiki_questions = ([], [])
    use_second = False
    for page in pages:
        for sentence in sentences_from_page(cw[page]):
            q = clean_question(sentence)
            wiki_questions[use_second].append(('-1', '1', page, q))
            use_second = not use_second

    vw_line = '{} {} \'{}|text {}\n'
    for i, wiki_qs in enumerate(wiki_questions):
        # Create list of True/False and shuffle to define ordering of train data
        order = list(chain(repeat(False, len(real_questions)), repeat(True, len(wiki_qs))))
        random.shuffle(order)
        iters = (iter(real_questions), iter(wiki_qs))
        with safe_open(DOMAIN_TARGET_PREFIX + str(i), 'w') as f:
            for choice in order:
                f.write(vw_line.format(*next(iters[choice])))


def _get_best(file_pairs, frac_questions):
    q_heap = []
    for (qs, scores) in file_pairs:
        for text_row, score_row in zip(qs, scores):
            label, _, page_and_namespace, text = text_row.split(' ', 3)
            if label == '1':
                # Don't add real questions
                continue
            # Page now has a leading single quote
            page, _ = page_and_namespace.split('|')
            page = page[1:]
            score = float(score_row.split(' ')[0])
            text = text.strip()
            heapq.heappush(q_heap, (score, text, page))

    n_questions = int(frac_questions * len(q_heap))
    while len(q_heap) > n_questions:
        heapq.heappop(q_heap)
    return [(t, p) for _, t, p in q_heap]


def get_best_wiki_questions(frac_questions=1.0):
    """Writes out a pickle containing a list of pairs of (text, page)"""
    log.info('Filtering down to top {}% of wikipedia sentences'.format(frac_questions * 100))
    with ExitStack() as stack:
        file_pairs = [(stack.enter_context(open(DOMAIN_TARGET_PREFIX + str(i))),
                       stack.enter_context(open(DOMAIN_PREDICTIONS_PREFIX + str(i))))
                      for i in (0, 1)]
        with safe_open(DOMAIN_OUTPUT.format('frac=' + str(frac_questions)), 'wb') as f:
            pickle.dump(_get_best(file_pairs, frac_questions), f)


if __name__ == '__main__':
    generate_domain_classifier_data()
