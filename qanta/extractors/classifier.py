from collections import defaultdict
from qanta.util.constants import CLASSIFIER_TYPES
from qanta.extractors.abstract import AbstractFeatureExtractor
from qanta.learning.classifier import load_classifier, load_guess_properties


class Classifier(AbstractFeatureExtractor):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifiers = {}
        class_to_i = defaultdict(dict)
        for c in CLASSIFIER_TYPES:
            self.classifiers[c] = load_classifier(c)
            for i, label in enumerate(self.classifiers[c].classes_):
                class_to_i[c][label] = i
        self.guess_properties = load_guess_properties()

    @property
    def name(self):
        return 'classifier'

    def score_guesses(self, guesses, text):
        zipped = {}
        for class_type, classifier in self.classifiers.items():
            probabilities = classifier.predict_proba([text])[0]
            classes = classifier.classes_
            zipped[class_type] = list(zip(classes, probabilities))

        for g in guesses:
            features = ['|classifier']
            for class_type in self.classifiers.keys():
                if g in self.guess_properties[class_type]:
                    prop = self.guess_properties[class_type][g]
                    for label, p in zipped[class_type]:
                        if label == prop:
                            features.append(
                                '{class_type}_{label}:{p}'.format(
                                    class_type=class_type,
                                    label=label,
                                    p=p
                                )
                            )
            yield ' '.join(features)
