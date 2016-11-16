import pickle

from qanta.preprocess import clean_question, replace_named_entities, format_guess
from qanta.util import qdb
from qanta.util.io import safe_open
from qanta.util.constants import MIN_APPEARANCES, DEEP_VOCAB_TARGET
from qanta.util.environment import QB_QUESTION_DB


class Preprocessor:
    def __init__(self):
        # map vocab to word embedding lookup index
        self.vocab = []
        self.vdict = {}

        # map stopwords to word embedding lookup index
        # todo: add "s" to stopwords
        self.stopset = set()

    def preprocess_input(self, q):
        q = clean_question(q)
        q = replace_named_entities(q)

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


def preprocess():
    pp = Preprocessor()
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
