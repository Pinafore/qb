import pickle
import argparse
import regex
import re

from qanta.util import qdb
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
            try:
                self.vdict[w]
            except:
                self.vocab.append(w)
                self.vdict[w] = len(self.vocab) - 1
            words.append(self.vdict[w])
        return words


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--threshold", type=int, default=5, help="Number of appearances")
    flags = parser.parse_args()

    pp = Preprocessor('data/internal/common/ners')
    db = qdb.QuestionDatabase(QB_QUESTION_DB)

    pages = set(db.page_by_count(min_count=flags.threshold))
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
                ans = q.page.strip().lower().replace(' ', '_')
                answer = pp.convert_to_indices(ans)
                proc_fold.append((qs, answer))
            if i % 5000 == 0:
                print('done with ', i)

        print(fold, len(proc_fold))
        pickle.dump(proc_fold, open('output/deep/' + fold, 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)

    pickle.dump((pp.vocab, pp.vdict), open('output/deep/vocab', 'wb'), \
                protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
