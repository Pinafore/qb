from contextlib import ExitStack
import heapq
from itertools import chain, repeat
import nltk
import pickle
import random

from qanta.guesser.util.preprocessing import preprocess_text
from qanta.util import qdb
from qanta.util.constants import DOMAIN_PREDICTIONS_PREFIX, DOMAIN_OUTPUT, DOMAIN_TARGET_PREFIX, MIN_APPEARANCES
from qanta.util.environment import QB_QUESTION_DB, QB_WIKI_LOCATION
from qanta.util.io import safe_open

from qanta.wikipedia.cached_wikipedia import CachedWikipedia


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
    db = qdb.QuestionDatabase(QB_QUESTION_DB)
    pages = set(db.page_by_count(min_count=MIN_APPEARANCES))
    cw = CachedWikipedia(QB_WIKI_LOCATION)
    qs = db.query('from questions where page != "" and fold == "train"', (), text=True)
    # Storing page with real questions doesn't matter
    real_questions = [('1', str(weight), 'real_question', preprocess_text(q.text[index])) for q in qs.values() for index in q.text]

    # Split wikipedia questions into two sets
    wiki_questions = ([], [])
    use_second = False
    for page in pages:
        for sentence in sentences_from_page(cw[page]):
            q = preprocess_text(sentence)
            wiki_questions[use_second].append(('-1', '1', page.strip().lower().replace(' ', '_'), q))
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


def _get_n_best(file_pairs, n_questions):
    best_questions = []
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
            heapq.heappush(best_questions, (score, text, page))
            if len(best_questions) > n_questions:
                heapq.heappop(best_questions)

    return [(t, p) for _, t, p in best_questions]


def get_best_wiki_questions(n_questions=5 * 10 ** 5):
    """Writes out a pickle containing a list of pairs of (text, page)"""
    with ExitStack() as stack:
        file_pairs = [(stack.enter_context(open(DOMAIN_TARGET_PREFIX + str(i))),
                       stack.enter_context(open(DOMAIN_PREDICTIONS_PREFIX + str(i))))
                      for i in (0, 1)]
        with safe_open(DOMAIN_OUTPUT, 'wb') as f:
            pickle.dump(_get_n_best(file_pairs, n_questions), f)

if __name__ == '__main__':
    generate_domain_classifier_data()
