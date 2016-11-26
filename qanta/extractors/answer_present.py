from math import log

from fuzzywuzzy import fuzz
from qanta.extractors.abstract import AbstractFeatureExtractor
from qanta.util.constants import ENGLISH_STOP_WORDS


class AnswerPresent(AbstractFeatureExtractor):
    @property
    def name(self):
        return 'answer_present'

    @staticmethod
    def score_one_guess(title, text):
        d = {}
        if "(" in title:
            title = title[:title.find("(")].strip()
        val = fuzz.partial_ratio(title, text)
        d["raw"] = log(val + 1)
        d["length"] = log(val * len(title) / 100. + 1)

        longest_match = 1
        for ii in title.split():
            if ii.lower() in ENGLISH_STOP_WORDS:
                continue
            longest_match = max(longest_match, len(ii) if ii in text else 0)
        d["longest"] = log(longest_match)

        return d

    def score_guesses(self, guesses, text):
        for guess in guesses:
            val = self.score_one_guess(guess, text)
            yield self.format_scores(val)

    def format_scores(self, results):
        return "|%s %s" % (self.name, " ".join("%s:%f" % (x, results[x]) for x in results))
