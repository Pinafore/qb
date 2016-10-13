import pickle
import regex
import re

from qanta.util import qdb
from qanta.util.io import safe_open
from qanta.util.constants import MIN_APPEARANCES, DEEP_VOCAB_TARGET
from qanta.util.environment import QB_QUESTION_DB


class Preprocessor:
    def __init__(self, ner_file):
        self.ners = pickle.load(open(ner_file, 'rb'))
        self.ftp = ["for 10 points, ", "for 10 points--", "for ten points, ", "for 10 points ",
                    "for ten points ", "ftp,", "ftp"]
        # map vocab to word embedding lookup index
        self.vocab = []
        self.vdict = {}

        # map stopwords to word embedding lookup index
        # todo: add "s" to stopwords
        self.stopset = set()

    def preprocess_input(self, q):
        q = q.strip().lower()

        # remove pronunciation guides and other formatting extras
        q = q.replace(' (*) ', ' ')
        q = q.replace('\n', '')
        q = q.replace('mt. ', 'mt ')
        q = q.replace(', for 10 points, ', ' ')
        q = q.replace(', for ten points, ', ' ')
        q = q.replace('--for 10 points--', ' ')
        q = q.replace(', ftp, ', ' ')
        q = q.replace('{', '')
        q = q.replace('}', '')
        q = q.replace('~', '')
        q = q.replace('(*)', '')
        q = q.replace('*', '')
        q = re.sub(r'\[.*?\]', '', q)
        q = re.sub(r'\(.*?\)', '', q)

        for phrase in self.ftp:
            q = q.replace(phrase, ' ')

        # remove punctuation
        q = regex.sub(r"\p{P}+", " ", q)

        # simple ner (replace answers w/ concatenated versions)
        for ner in self.ners:
            q = q.replace(ner, ner.replace(' ', '_'))

        words = self.convert_to_indices(q.strip())
        return words

    def convert_to_indices(self, text):
        words = []
        for w in text.split():
            if w not in self.vdict:
                self.vocab.append(w)
                self.vdict[w] = len(self.vocab) - 1
            words.append(self.vdict[w])
        return words


def format_guess(guess):
    return guess.strip().lower().replace(' ', '_')


def preprocess():
    pp = Preprocessor('data/internal/common/ners')
    db = qdb.QuestionDatabase(QB_QUESTION_DB)

    pages = set(db.page_by_count(min_count=MIN_APPEARANCES))
    print(len(pages))
    folds = ['train', 'test', 'devtest', 'dev']
    for fold in folds:
        allqs = db.query('from questions where page != "" and fold == ?', (fold,), text=True)
        print(fold, len(allqs))
        proc_fold = []
        for i, key in enumerate(allqs):
            q = allqs[key]
            if q.page in pages:
                qs = {}
                for index in q.text:
                    qs[index] = pp.preprocess_input(q.text[index])
                ans = format_guess(q.page)
                answer = pp.convert_to_indices(ans)
                proc_fold.append((qs, answer))
            if i % 5000 == 0:
                print('done with ', i)

        print(fold, len(proc_fold))
        with safe_open('output/deep/' + fold, 'wb') as f:
            pickle.dump(proc_fold, f, protocol=pickle.HIGHEST_PROTOCOL)

    with safe_open(DEEP_VOCAB_TARGET, 'wb') as f:
        pickle.dump((pp.vocab, pp.vdict), f, protocol=pickle.HIGHEST_PROTOCOL)
