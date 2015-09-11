import sys
import pickle
import datetime
import operator
import codecs
import random
from unidecode import unidecode
from glob import glob
from collections import defaultdict
from random import shuffle, sample

from sklearn import linear_model
import pandas as pd

kDERIVED_COLUMNS = [('Astronomy', ''), ('Biology', ''), ('Chemistry',
''), ('Chemistry', 'chemistry)'), ('Earth Science', ''), ('Fine Arts',
''), ('History', ''), ('Mathematics', ''), ('Other', ''), ('Physics',
''), ('Social Studies', ''), ('Literature', ''), ('Literature',
'film)'), ('Literature', 'novel)'), ('Literature', 'play)'),
('Literature', 'mythology)'), ('Literature', 'miniseries)'),
('Literature', 'essay)'), ('Literature', 'opera)'), ('Literature',
'story)'), ('Literature', 'poem)'), ('Literature', 'serial)'),
('Literature', 'manga)'), ('Literature', 'constellation)'),
('Literature', 'musical)')]

def load_mapping(match_filename):
    res = {}
    for ii in glob("%s*" % match_filename):
        for jj in codecs.open(ii, 'r', 'utf-8'):
            if jj.strip() == "":
                continue

            try:
                id, page = jj.split("\t", 1)
                id = int(id)
            except ValueError:
                page = ""
                id = int(jj)
            res[id] = page.strip()
    return res

def paren_match(row):
    start = row['page'].find('(')
    stop = row['page'].find(')', start)

    if stop > 0 and start > 0 and \
      row['page'][(start + 1):(stop - 1)].lower() in row['text'].lower():
      return 1.0
    else:
        return 0.0

class ActiveLearner:
    def read_data(self, filename, labels, max_size=0):
        """
        Read a new cvs file as data frame, discarding data if it's too
        large.
        """
        if max_size > 0:
            print("Reading %i rows of %s" % (max_size, filename))
            new_data = pd.read_csv(filename, nrows=max_size)
        else:
            print("Reading all rows of %s" % filename)
            new_data = pd.read_csv(filename)
        return new_data

    def __init__(self, input_csv, match_filename, train_columns=[], max_size=-1):
        self._raw = pd.DataFrame()
        self._match_filename = match_filename
        self._current_matches = {}
        self._guess = None
        self._positive = pd.DataFrame()
        self._negative = pd.DataFrame()
        self._train_cols = train_columns
        self._stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

        # Read in our labels
        print("Searching for " + "%s.*" % match_filename)
        self._old_matches = load_mapping(match_filename)
        print("Reading %i" % max_size)

        # Read in features
        if input_csv:
            for ii in input_csv:
                new_data = self.read_data(ii, self._old_matches, max_size)
                self._raw = self._raw.append(new_data)
            self._raw = self.add_derived_columns(self._raw)
            for ii in self._old_matches:
                self.add_training(ii, self._old_matches[ii])

            self.clean_data()

    def clean_data(self):
        # Clean up errors
        for ii in self._train_cols:
            try:
                if max(self._raw[ii]) == 'None':
                    self._raw[self._raw[ii]=='None'] = 0
                    self._raw[ii] = self._raw[ii].astype(float)
            except NameError:
                continue
            except KeyError:
                continue

        # Remove features that don't appear in training data
        to_remove = []
        for ii in self._train_cols:
            if len(self._positive) == 0 or \
              min(min(self._positive[ii]), min(self._negative[ii])) == \
              max(max(self._positive[ii]), max(self._negative[ii])):
              to_remove.append(ii)
        for ii in to_remove:
            if ii != 'bias':
                self._train_cols.remove(ii)

    def add_derived_columns(self, dataset):
        self._raw['bias'] = 1
        print("Adding derived columns:")
        for cat, pattern in kDERIVED_COLUMNS:
            col_name = "%s-%s" % (cat, pattern[:-1])
            print("\t%s" % col_name)
            dataset[col_name] = dataset.apply(lambda row: 1.0 if \
                                              pattern in str(row['page']).lower() and \
                                              row['category'] == cat else 0.0, axis=1)
            dataset[col_name] = dataset[col_name].astype(float)
            if not col_name in self._train_cols:
                self._train_cols.append(col_name)

        self._raw['paren_match'] = self._raw.apply(paren_match, axis=1).astype(float)
        if not 'paren_match' in self._train_cols:
            self._train_cols.append('paren_match')

        return dataset

    def human_labeled(self):
        for ii in self._old_matches:
            yield ii, self._old_matches[ii]

    def remember(self, id, wiki_page):
        if len(self._raw) > 0:
            self.add_training(id, wiki_page)
        self._current_matches[id] = wiki_page

    def dump(self, filename):
        outfile = "%s_%s" % (filename, self._stamp)
        print("Writing to %s" % outfile)
        o = codecs.open(outfile, 'w', 'utf-8')
        for ii in self._current_matches:
            o.write("%i\t%s\n" % (int(ii), self._current_matches[ii]))

    def relearn(self):
        assert len(self._positive) > 0, "We can't learn on empty data (check the input CSVs)"
        print("Training columns: ", " ".join(self._train_cols))
        train = self._positive.append(self._negative)
        print("Training (%i) ... " % len(train))

        logreg = linear_model.LogisticRegression(C=1e5)
        logreg.fit(train[self._train_cols], train['corr'])

        print("-------------------------")
        for nn, vv in zip(self._train_cols, logreg.coef_[0]):
            print("\t%30s, %f" % (nn, vv))
        print("-------------------------")

        self._guess = logreg.predict_proba(self._raw[self._train_cols])
        self._raw['guess'] = self._guess[:,1]

        print("Total predictions: %i" % len(self._raw))

    def predictions(self, min_pos=0.9):
        if self._guess != None:
            possible_answers = set(int(x) for x in set(self._raw['id']))
            for ii in possible_answers:
                if ii in self._old_matches:
                    yield ii, self._old_matches[ii], 1.0
                else:
                    this_guess = max(self._raw[self._raw['id']==ii]['guess'])
                    answer = self._raw[(self._raw['id']==ii) & (self._raw['guess']==this_guess)]
                    answer = max(answer['page'])
                    if this_guess > min_pos:
                        yield ii, answer, this_guess

    def uncertain(self, count=10):
        possible_answers = set(int(x) for x in set(self._raw['id']))
        possible_answers = possible_answers - set(int(x) for x in self._negative['id'])

        # If we have logistic predictions, we can find the most
        # confused guess, otherwise, just use a single column
        if self._guess != None:
            comparison_col = 'guess'
        else:
            comparison_col = 'body_score'

        values = {'id': possible_answers}
        row_mask = self._raw.isin(values).any(1)
        possible_guesses = self._raw[row_mask]
        possible_guesses['id'] = possible_guesses['id'].astype(int)

        # Get the best guess for each answer
        guesses = defaultdict(float)
        for num, row in possible_guesses.iterrows():
            guesses[row['id']] = max(guesses[row['id']], row[comparison_col])

        # 0 as a default value sometimes screws us up
        # TODO(jbg): Fix the real source of this issue, wherever it is
        if 0 in guesses:
            del guesses[0]

        guesses = sorted(guesses.iteritems(), key=operator.itemgetter(1))[:count]

        for qq, score in guesses:
            yield qq, comparison_col

    def add_training(self, id, wiki_page):
        # Add positive examples
        guesses = self._raw[self._raw['id'] == id]

        correct = guesses[guesses['page'] == wiki_page]
        wrong = guesses[guesses['page'] != wiki_page]

        self._positive = self._positive.append(correct)
        self._negative = self._negative.append(wrong)

        # Add column (or replace exiting column)
        self._positive['corr'] = 1
        self._negative['corr'] = 0

    def train_classifier(self):
        full_data = self._positive.append(self._negative)
        logit = sm.Logit(full_data['corr'], full_data[self._train_cols])
        logit.raise_on_perfect_prediction = False
        result = logit.fit()

        self._guess = result.predict(self._raw[self._train_cols])


def simple_menu(choices, index, scores=None, escape='x'):
    """
    Given two lists of choices and scores, present menu with the two
    of them.
    """

    assert scores is None or len(scores) == len(choices), \
      "Score must have same length as choices got %i vs %i" % (len(scores), len(choices))
    if scores is None:
        scores = [float("-inf")] * len(choices)

    chosen_page = None
    while chosen_page is None:
        print("---------------------------------------------------")
        for ii, (choice, score) in enumerate(zip(choices, scores)):
            if score > float("-inf"):
                print("%i)\t%s\t%0.2f" % (ii, choice, score))
            else:
                print("%i)\t%s" % (ii, choice))
        usr = raw_input("Enter wikipedia page:").decode(sys.stdin.encoding)
        usr = usr.replace("_", " ").strip()

        try:
            if int(usr) in xrange(len(choices)):
                chosen_page = choices[int(usr)]
                print("Good choice, %i: %s" % (int(usr), chosen_page))
        except ValueError:
            if usr.startswith("!"):
                chosen_page = usr[1:]
            elif usr != "" and not usr in index:
                print("Nope, not found; try again")
                chosen_page = None
            else:
                chosen_page = usr

        print(chosen_page)
        if usr.lower() == escape:
            break

    return chosen_page


# if __name__ == "__main__":
#     flags.define_glob("raw_csv", "data/*.csv", "Input file")
#     flags.define_int("max_size", -1, "Max size of our raw dataset")
#     flags.define_string("wiki_index", None, "Index of wikipages")
#     flags.define_list("train_columns", ["title_score", "title_edit_dist", "bias", "body_score"],
#                       "Columns used to build model")
#     flags.define_string("match_location", None,
#                         "Where we write the matches learned")

#     flags.InitFlags()

#     al = ActiveLearner(flags.raw_csv, flags.match_location, flags.train_columns, max_size=flags.max_size)
#     al.relearn()

#     if flags.wiki_index:
#         wiki_index = pickle.load(open(flags.wiki_index))
#     else:
#         wiki_index = []

#     interactions = 0
#     usr = ''
#     for qid, column in al.uncertain():
#         candidates = al._raw[al._raw['id'] == qid].sort(column, ascending=False)
#         choices = candidates['page']
#         scores = candidates['guess']

#         print(max(candidates['text']))
#         print(max(candidates['answer']))
#         chosen_page = simple_menu(choices, wiki_index, scores, 'x')

#         if chosen_page is not None:
#             al.remember(qid, chosen_page)
#             al.relearn()
#         else:
#             break

#     al.dump(flags.match_location)
