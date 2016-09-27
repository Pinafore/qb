import pickle

from qanta.util.constants import CLASSIFIER_PICKLE_PATH, CLASSIFIER_TYPES
from qanta.extractors.abstract import FeatureExtractor


class Classifier(FeatureExtractor):
    def __init__(self, question_db):
        super(Classifier, self).__init__()
        self.qdb = question_db
        self.classifiers = {}
        self.name = 'classifier'
        for c in CLASSIFIER_TYPES:
            self.add_classifier(c)

    def add_classifier(self, classifier_type):
        classifier_path = CLASSIFIER_PICKLE_PATH.format(classifier_type)
        with open(classifier_path, 'rb') as f:
            self.classifiers[classifier_type] = pickle.load(f)

    def score_guesses(self, guesses, text):
        for guess in guesses:
            self.featurize(text)
            # majority = self.majority(title)

            val = ["|classifier"]
            for class_type, classifier in self.classifiers.items():
                val.append("{class_type}_maj:{prob}".format(
                    class_type=class_type, prob=0))

            yield ' '.join(val), guess

    def featurize(self, text):
        pass
