try:
    import cPickle as pickle
except:
    import pickle

import re
from feature_extractor import FeatureExtractor
from unidecode import unidecode
from collections import defaultdict, Counter
from nltk.util import ngrams

alphanum = re.compile('[\W_]+')
kCLASSIFIER_FIELDS = ["category", "ans_type", "gender"]


class Classifier(FeatureExtractor):
    def __init__(self, bigram_path, question_db):
        self._qdb = question_db
        self._bigrams = pickle.load(open(bigram_path, 'rb'))
        self._majority = {}
        self._frequencies = defaultdict(dict)
        self._cache = None
        self._fv = None
        self._pd = None
        self._classifiers = {}
        self._name = 'classifier'
        self.add_classifier('data/classifier/category.pkl', 'category')
        self.add_classifier('data/classifier/ans_type.pkl', 'ans_type')
        self.add_classifier('data/classifier/gender.pkl', 'gender')
        self.cache_majorities('category')
        self.cache_majorities('ans_type')
        self.cache_majorities('gender')

    def cache_majorities(self, attribute):
        self._majority[attribute] = defaultdict(Counter)
        all_questions = self._qdb.questions_with_pages()
        for page in all_questions:
            for qq in all_questions[page]:
                if qq.fold == 'train':
                    self._majority[attribute][qq.page][getattr(qq, attribute, "").split(":")[0].lower()] += 1

        # normalize counter
        for page in self._majority[attribute]:
            self._majority[attribute][page] = \
                self._majority[attribute][page].most_common(1)[0][0]
            # total = sum(self._majority[attribute][page].values(), 0.0)
            # for key in self._majority[attribute][page]:
            #     self._majority[attribute][page][key] /= total

    def add_classifier(self, classifier_path, column):
        self._classifiers[column] = pickle.load(open(classifier_path, 'rb'))

    def vw_from_title(self, title, text):
        pd = self.featurize(text)
        # majority = self.majority(title)

        val = ["|classifier"]
        for cc in self._classifiers:
            majority = self._majority[cc][title]
            pd = self._pd[cc]
            # for ii in pd.samples():
            #     val.append("%s_%s:%f" % (cc, ii, pd.prob(ii)))
            try:
                val.append("%s_maj:%f" % (cc, pd.prob(majority)))
            except:
                pass
            # val.append("%s_wmaj:%f" % (cc, pd.prob(majority[cc][0]) *
            #                            pd.prob(majority[cc][1])))

        return ' '.join(val)

    def featurize(self, text):
        if hash(text) != self._cache:
            self._cache = hash(text)
            feats = {}
            total = alphanum.sub(' ', unidecode(text.lower()))
            total = total.split()
            bgs = set(ngrams(total, 2))
            for bg in bgs.intersection(self._bigrams):
                feats[bg] = 1.0
            for word in total:
                feats[word] = 1.0
            # self._fv = feats
            self._pd = {}
            for cc in self._classifiers:
                self._pd[cc] = self._classifiers[cc].prob_classify(feats)
        return self._pd

    def majority(self, guess):
        if not guess in self._majority:
            for cc in self._classifiers:
                self._majority[guess][cc] = self._qdb.majority_frequency(guess, cc)
        return self._majority[guess]




if __name__ == "__main__":
    pass
