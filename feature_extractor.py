class FeatureExtractor:
    def guesses(self, question):
        """
        Returns all of the guesses for a given question.  If this depends on
        another system for generating guesses, it can return an empty list.
        """

        raise NotImplementedError

    @staticmethod
    def has_guess():
        return False

    def namespace(self):
        return self._name

    def features(self, question, candidate):
        """
        Given a question and a candidate, returns the features
        """
        raise NotImplementedError

    def set_metadata(self, answer, category, qnum, sent, token, guesses, fold):
        self._correct = answer
        self._num_guesses = guesses
        self._qnum = qnum
        self._sent = sent
        self._token = token
        self._fold = fold
        self._id = '%i_%i_%i' % (self._qnum, self._sent, self._token)

    def set_num_guesses(self, num_guesses):
        None

    def vw_from_score(self, results):
        """
        Dictionary of feature key / value pairs
        """
        raise NotImplementedError

    def vw_from_title(self, title, text):
        raise NotImplementedError

    def name(self):
        return self._name
