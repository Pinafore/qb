import pickle

from qanta.guesser.util import preprocessing

from qanta.util import qdb
from qanta.util.io import safe_open
from qanta.util.constants import DEEP_VOCAB_TARGET, DEEP_WIKI_TARGET, DOMAIN_OUTPUT, MIN_APPEARANCES, NERS_LOCATION
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

    def preprocess_input(self, q, add_unseen_words=True):
        q = preprocessing.preprocess_text(q, ners=self.ners)
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
    pp = Preprocessor(NERS_LOCATION)
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
                ans = preprocessing.preprocess_answer(q.page.strip().lower().replace(' ', '_'))
                answer = pp.convert_to_indices(ans)
                proc_fold.append((qs, answer))
            if i % 5000 == 0:
                print('done with ', i)

        print(fold, len(proc_fold))
        with safe_open('output/deep/' + fold, 'wb') as f:
            pickle.dump(proc_fold, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(DOMAIN_OUTPUT, 'rb') as f, open(DEEP_WIKI_TARGET, 'wb') as out:
        processed = [({0: pp.convert_to_indices(text)}, pp.convert_to_indices(page)) for text, page in pickle.load(f)]
        pickle.dump(processed, out, protocol=pickle.HIGHEST_PROTOCOL)

    with safe_open(DEEP_VOCAB_TARGET, 'wb') as f:
        pickle.dump((pp.vocab, pp.vdict), f, protocol=pickle.HIGHEST_PROTOCOL)
