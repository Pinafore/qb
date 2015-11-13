from __future__ import absolute_import
import abc


class FeatureExtractor(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._correct = None
        self._num_guesses = None
        self._qnum = None
        self._sent = None
        self._token = None
        self._fold = None
        self._id = None
        self.name = None

    @abc.abstractmethod
    def guesses(self, question):
        """
        Returns all of the guesses for a given question.  If this depends on
        another system for generating guesses, it can return an empty list.
        """
        pass

    @staticmethod
    def has_guess():
        return False

    @abc.abstractmethod
    def features(self, question, candidate):
        """
        Given a question and a candidate, returns the features
        """
        pass

    def set_metadata(self, answer, category, qnum, sent, token, guesses, fold):
        self._correct = answer
        self._num_guesses = guesses
        self._qnum = qnum
        self._sent = sent
        self._token = token
        self._fold = fold
        self._id = '%i_%i_%i' % (self._qnum, self._sent, self._token)

    def set_num_guesses(self, num_guesses):
        pass

    @abc.abstractmethod
    def vw_from_score(self, results):
        """
        Dictionary of feature key / value pairs
        """
        pass

    @abc.abstractmethod
    def vw_from_title(self, title, text):
        pass
