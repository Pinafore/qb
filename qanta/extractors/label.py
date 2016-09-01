import numpy as np
from unidecode import unidecode
from qanta.extractors.abstract import FeatureExtractor


class Labeler(FeatureExtractor):
    def __init__(self, question_db):
        super(Labeler, self).__init__()
        self.name = 'label'
        self.counts = {}
        all_questions = question_db.questions_with_pages()

        # Get the counts
        for ii in all_questions:
            self.counts[ii] = sum(1 for x in all_questions[ii] if
                                   x.fold == "train")
        # Standardize the scores
        count_mean = np.mean(list(self.counts.values()))
        count_var = np.var(list(self.counts.values()))
        for ii in all_questions:
            self.counts[ii] = float(self.counts[ii] - count_mean) / count_var

    def score_guesses(self, guesses, text):
        for guess in guesses:
            formatted_guess = guess.replace(":", "").replace("|", "")

            # TODO: Incorporate token position here as well to improve
            # position-based features
            if formatted_guess == self._correct:
                feature = "1 '%s |guess %s sent:%0.1f count:%f" % \
                    (self._id, unidecode(formatted_guess).replace(" ", "_"),
                     self._sent, self.counts.get(formatted_guess, -2))
                yield feature, guess
            else:
                feature = "-1 %i '%s |guess %s sent:%0.1f count:%f" % \
                    (self._num_guesses, self._id,
                     unidecode(formatted_guess).replace(" ", "_"),
                     self._sent, self.counts.get(formatted_guess, -2))
                yield feature, guess
