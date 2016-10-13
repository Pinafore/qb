import numpy as np
from unidecode import unidecode
from functional import seq
from qanta.extractors.abstract import FeatureExtractor
from qanta.util.qdb import QuestionDatabase


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
        n_words = len(text.split())
        for guess in guesses:
            formatted_guess = guess.replace(":", "").replace("|", "")

            if formatted_guess == self._correct:
                feature = "1 '%s |guess %s sent:%0.1f count:%f words_seen:%i" % \
                    (self._id, unidecode(formatted_guess).replace(" ", "_"),
                     self._sent, self.counts.get(formatted_guess, -2), n_words)
                yield feature, guess
            else:
                feature = "-1 %i '%s |guess %s sent:%0.1f count:%f words_seen:%i" % \
                    (self._num_guesses, self._id,
                     unidecode(formatted_guess).replace(" ", "_"),
                     self._sent, self.counts.get(formatted_guess, -2), n_words)
                yield feature, guess


def compute_question_stats(question_db: QuestionDatabase):
    questions = [q for q in question_db.guess_questions() if q.fold == 'train' or q.fold == 'dev']
    sentence_lengths = seq(questions)\
        .flat_map(lambda q: q.text.values()).map(lambda q: len(q.split())).list()
    mean_sentence = np.mean(sentence_lengths)
