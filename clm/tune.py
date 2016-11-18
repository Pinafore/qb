# Script to generate different configurations of LM, train an sklearn
# classifier, then compare the results

from lm_wrapper import LanguageModelReader, LanguageModelBase, pretty_debug
from csv import DictReader
from collections import defaultdict
from math import floor

          # feature: start, min, max, increment
kFEATS = {"interp": (0.8, 0.0, 1.0, 0.05),
          "min_span": (2, 1, 4, 1),
          "max_span": (5, 2, 10, 1),
          "start_rank": (200, 0, 1000, 50),
          "smooth": (0.01, 0.001, 1.0, 0.001),
          "cutoff": (-2, -100, 0, 1),
          "slop": (0, 0, 2, 1),
          "censor_slop": (True, False, True, "bool"),
          "log_length": (True, False, True, "bool"),          
          "give_score": (True, False, True, "bool")}

class Parameters:
    def __init__(self, param_constraints):
        for ff in kFEATS:
            start, low, high, delta = param_constraints[ff]

            self._min[ff] = low
            self._max[ff] = high
            self._delta[ff] = delta

            self._vals[ff] = start

    def propose_new(self, feature):
        current = self._vals[feature]

        delta = self._delta[feature]
        if delta == "bool"
            return not current
        elif isinstance(delta, int):
            return floor(current + (2 * delta) * (0.5 - random()) + 1)
                        
    def set_params(self, lm):
        lm.set_params(interp = self._vals["interp"],
                      min_span = self._vals["min_span"],
                      max_span = self._vals["max_span"],
                      start_rank = self._vals["start_rank"],
                      smooth = self._vals["smooth"],
                      cutoff = self._vals["cutoff"],
                      slop = self._vals["slop"],
                      censor_slop = self._vals["censor_slop"],
                      give_score = self._vals["give_score"],
                      log_score = self._vals["log_score"])

def build_dataset(lm, db, min_appearances, restrict_frequent=True, seed=1701):
    # Get page names
    pages = db.page_by_count(min_appearances)
    
    # random train / test split
    
        
def score(lm, train, test):
    # generate features from the dataset

    # train a classifier on train fold

    # apply it to the test fold

    # compute the accuracy and return that number

def logistic(x):
    return exp(x) / (1 + exp(x))
    
if __name__ == "__main__":
    lm = LanguageModelReader("data/language_model")
    lm.init()

    old_score = 0.0
    temperature = 1.0
    params = Parameters(kFEATS)
    for ii in range(flags.num_passes):
        for ff in kFEATS:
            old = params.propose_new(ff)
            proposed = params.propose_new(ff)
            params.set_params(lm)

            new_score = score(lm, train, test)

            temperature *= flags.schedule

            if logistic(new_score) * temperature > random()
    
