import pandas as pd

from qanta.util.constants import CLASSIFIER_TYPES
from qanta.extractors.abstract import FeatureExtractor
from qanta.learning.classifier import load_classifier


class Classifier(FeatureExtractor):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifiers = {}
        self.name = 'classifier'
        for c in CLASSIFIER_TYPES:
            self.classifiers[c] = load_classifier(c)

    def score_guesses(self, guesses, text):
        df = pd.DataFrame({'text': pd.Series([text])})

        features = ['|classifier']
        for class_type, classifier in self.classifiers.items():
            probabilities = classifier.predict_proba(df)
            if len(probabilities) == 0:
                for label in classifier.classes_:
                    features.append(
                        '{class_type}_{label}:{p}'.format(class_type=class_type, label=label, p=-1)
                    )
            else:
                for label, p in zip(classifier.classes_, classifier.predict_proba(df)[0]):
                    features.append(
                        '{class_type}_{label}:{p}'.format(class_type=class_type, label=label, p=p)
                    )

        feature_string = ' '.join(features)

        for guess in guesses:
            yield feature_string, guess
