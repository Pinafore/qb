import heapq
from itertools import chain, repeat
import nltk
import pickle
import random

from qanta.guesser.util.preprocessing import preprocess_text
from qanta.util import qdb
from qanta.util.constants import NERS_LOCATION,  DOMAIN_CLASSIFIER_TARGET_PREFIX
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
    pages = set(db.page_by_count())
    cw = CachedWikipedia(QB_WIKI_LOCATION)
    qs = db.query('from questions where page != "" and fold == "train"', (), text=True)
    real_questions = [('1', str(weight), preprocess_text(q.text[index])) for q in qs.values() for index in q.text]

    # Split wikipedia questions into two sets
    wiki_questions = ([], [])
    use_second = False
    for page in pages:
        for sentence in sentences_from_page(cw[page]):
            q = preprocess_text(sentence)
            wiki_questions[use_second].append(('-1', '1', q))
            use_second = not use_second
            if len(wiki_questions) % 10000 == 0:
                print("Loaded {} sentences from wikipedia".format(len(data)))

    vw_line = '{} {} |text {}\n'
    for i, wiki_qs in enumerate(wiki_questions):
        # Create list of True/False and shuffle to define ordering of train data
        order = list(chain(repeat(False, len(real_questions)), repeat(True, len(wiki_qs))))
        random.shuffle(order)
        iters = (iter(real_questions), iter(wiki_qs))
        with open(DOMAIN_CLASSIFIER_TARGET_PREFIX + str(i), 'w') as f:
            for choice in order:
                f.write(vw_line.format(*next(iters[choice])))

def get_best_wiki_questions(file_pairs, n_questions=5*10**5):
    best_questions = []
    for (qs, scores) in file_pairs:
        for text_row, score_row in zip(text_row, score_row):
            q_id = something(text_row)
            score = float(score_row.strip())
            heapq.heappush(best_questions, (score, q_id))
            if len(best_questions) > n_questions:
                heapq.heappop(best_questions)

    return [i for _, i in best_questions]

if __name__ == '__main__':
    generate_wikipedia_questions()
