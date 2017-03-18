from qanta.util.constants import CLASSIFIER_TYPES
from qanta.extractors.abstract import AbstractFeatureExtractor
from qanta.learning.classifier import load_classifier


class Classifier(AbstractFeatureExtractor):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifiers = {}
        for c in CLASSIFIER_TYPES:
            self.classifiers[c] = load_classifier(c)

    @property
    def name(self):
        return 'classifier'

    def score_guesses(self, guesses, text):
        features = ['|classifier']
        for class_type, classifier in self.classifiers.items():
            probabilities = classifier.predict_proba(text)
            if len(probabilities) == 0:
                for label in classifier.classes_:
                    features.append(
                        '{class_type}_{label}:{p}'.format(class_type=class_type, label=label, p=-1)
                    )
            else:
                for label, p in zip(classifier.classes_, probabilities[0]):
                    features.append(
                        '{class_type}_{label}:{p}'.format(class_type=class_type, label=label, p=p)
                    )

        feature_string = ' '.join(features)

        for _ in guesses:
            yield feature_string
