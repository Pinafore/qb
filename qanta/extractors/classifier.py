from unidecode import unidecode
import pickle
from collections import defaultdict, Counter
from nltk.util import ngrams

from qanta.util.constants import ALPHANUMERIC
from qanta.extractors.abstract import FeatureExtractor


class Classifier(FeatureExtractor):
    def __init__(self, question_db):
        super(Classifier, self).__init__()
        self.qdb = question_db
        with open('output/classifier/category/bigrams.pkl', 'rb') as f:
            self.bigrams = pickle.load(f)
        self.majority = {}
        self.frequencies = defaultdict(dict)
        self.cache = None
        self.fv = None
        self.pd = None
        self.classifiers = {}
        self.name = 'classifier'
        self.add_classifier('output/classifier/category.pkl', 'category')
        self.add_classifier('output/classifier/ans_type.pkl', 'ans_type')
        self.add_classifier('output/classifier/gender.pkl', 'gender')
        self.cache_majorities('category')
        self.cache_majorities('ans_type')
        self.cache_majorities('gender')

    def set_metadata(self, answer, category, qnum, sent, token, guesses, fold):
        pass

    def cache_majorities(self, attribute):
        self.majority[attribute] = defaultdict(Counter)
        all_questions = self.qdb.questions_with_pages()
        for page in all_questions:
            for qq in all_questions[page]:
                if qq.fold == 'train':
                    self.majority[attribute][qq.page][
                        getattr(qq, attribute, "").split(":")[0].lower()] += 1

        # normalize counter
        for page in self.majority[attribute]:
            self.majority[attribute][page] = \
                self.majority[attribute][page].most_common(1)[0][0]
            # total = sum(self._majority[attribute][page].values(), 0.0)
            # for key in self._majority[attribute][page]:
            #     self._majority[attribute][page][key] /= total

    def add_classifier(self, classifier_path, column):
        with open(classifier_path, 'rb') as f:
            self.classifiers[column] = pickle.load(f)

    def vw_from_title(self, title, text):
        self.featurize(text)
        # majority = self.majority(title)

        val = ["|classifier"]
        for cc in self.classifiers:
            majority = self.majority[cc][title]
            pd = self.pd[cc]
            val.append("%s_maj:%f" % (cc, pd.prob(majority)))

        return ' '.join(val)

    def featurize(self, text):
        if hash(text) != self.cache:
            self.cache = hash(text)
            feats = {}
            total = ALPHANUMERIC.sub(' ', unidecode(text.lower()))
            total = total.split()
            bgs = set(map(str, ngrams(total, 2)))
            for bg in bgs.intersection(self.bigrams):
                feats[bg] = 1.0
            for word in total:
                feats[word] = 1.0
            # self._fv = feats
            self.pd = {}
            for cc in self.classifiers:
                self.pd[cc] = self.classifiers[cc].prob_classify(feats)
        return self.pd

    def majority(self, guess):
        if guess not in self.majority:
            for cc in self.classifiers:
                self.majority[guess][cc] = self.qdb.majority_frequency(guess, cc)
        return self.majority[guess]

    def vw_from_score(self, results):
        pass
