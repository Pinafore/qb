import abc


class FeatureExtractor(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._correct = None
        self._num_guesses = None
        self._qnum = 0
        self._sent = 1
        self._token = 0
        self._fold = None
        self._id = '0'
        self.name = None

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
    def vw_from_title(self, title: str, text: str):
        pass
