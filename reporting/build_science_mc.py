# Script to generate output equivalent to the AI2 Kaggle science challenge

import sqlite3
import operator
import random
from csv import DictWriter
from collections import defaultdict
from extract_features import instantiate_feature, guesses_for_question
from util.qdb import QuestionDatabase

from unidecode import unidecode

kCOUNT_CUTOFF = 2
kCHOICEIDS = "ABCDEFGHIJKLMNOP"

kCATEGORIES = set("""Science
Science:Astronomy
Science:Biology
Science:Chemistry
Science:Computer_Science
Science:Earth_Science
Science:Math
Science:Mathematics
Science:Other
Mathematics
Physics
Biology
Chemistry
Earth Science
Science:Physics""".split("\n"))

class McScience:
    def __init__(self, page, question, fold):
        self.page = page
        self.question = question
        self.choices = []
        self.fold = fold

    def add_text(self, text):
        self.text = text

    def add_choices(self, choices):
        self.choices = list(choices)
        random.shuffle(self.choices)

    def csv_line(self, choice_strings, destination="train"):
        d = {}
        d["id"] = self.question
        if destination != "key":
            d["question"] = self.text

        for ii, cc in enumerate(self.choices):
            if destination != "key":
                d["answer%s" % choice_strings[ii]] = unidecode(cc)
            if cc == self.page and (destination == "train" or destination == "key"):
                d["correctAnswer"] = choice_strings[ii]

        assert self.page in self.choices, "Correct answer %s not in the set %s" % \
            (self.page, str(self.choices))
        return d

def question_top_guesses(text, deep, guess_connection, id, page, num_guesses=4):
    """
    Return the top guesses for this page
    """

    c = guess_connection.cursor()
    command = ('select page from guesses where sentence = 2 and token = 0 and question = %i ' +
               'order by score desc limit %i') % (id, num_guesses+1)
    c.execute(command)

    choices = set([page])
    for ii, in c:
        if len(choices) < num_guesses and not ii in choices:
            choices.add(ii)

    # If we don't have enough guesses, generate more
    new_guesses = deep.text_guess(text)

    # sort the guesses and add them
    for guess, score in sorted(new_guesses.items(), key=operator.itemgetter(1), reverse=True):
        if len(choices) < num_guesses and not guess in choices:
            choices.add(guess)

    return choices

def question_first_sentence(database_connection, question):
    """
    return the id, answer, and first sentence of questions in a set of categories
    """
    c = database_connection.cursor()
    command = 'select raw from text where question=%i' % question
    c.execute(command)

    for ii, in c:
        return unidecode(ii)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    default_path = 'data/'
    parser.add_argument('--question_db', type=str, default=default_path + 'questions.db')
    parser.add_argument('--guess_db', type=str, default=default_path + 'guesses.db',
                        help="Guess database")
    parser.add_argument("--num_choices", type=int, default=4,
                        help="How many choices do we write")
    parser.add_argument("--train_out", type=str, default="sci_train.csv")
    parser.add_argument("--test_out", type=str, default="sci_test.csv")
    parser.add_argument("--key_out", type=str, default="sci_key.csv")
    flags = parser.parse_args()

    # Create database connections
    print("Opening %s" % flags.question_db)
    question_database = sqlite3.connect(flags.question_db)
    guess_database = sqlite3.connect(flags.guess_db)

    # First get answers of interest and put them in a dictionary where the value is their count
    query = 'select page from questions where page != "" and ('
    query += " or ".join("category='%s'" % x for x in kCATEGORIES)
    query += ")"
    c = question_database.cursor()
    print(query)
    c.execute(query)

    answer_count = defaultdict(int)
    for pp, in c:
        answer_count[pp] += 1

    query = 'select page, id, naqt, fold from questions where page != ""'
    c = question_database.cursor()
    c.execute(query)

    print(list(x for x in answer_count if answer_count[x] >= kCOUNT_CUTOFF))
    print(len(list(x for x in answer_count if answer_count[x] >= kCOUNT_CUTOFF)))

    # Load the DAN to generate guesses if they're missing from the database
    deep = instantiate_feature("deep", QuestionDatabase(flags.question_db))

    questions = {}
    question_num = 0
    for pp, ii, nn, ff in c:
        if nn >= 0 or answer_count[pp] < kCOUNT_CUTOFF:
            continue
        question_num += 1
        question = McScience(pp, ii, ff)
        question.add_text(question_first_sentence(question_database, ii))
        choices = question_top_guesses(question.text, deep, guess_database, ii, pp,
                                       flags.num_choices)
        question.add_choices(choices)
        questions[ii] = question
        if question_num % 100 == 0:
            print(pp, ii, question_num)
            print(choices)

    answer_choices = ["answer%s" % kCHOICEIDS[x] for x in xrange(flags.num_choices)]

    train_out = DictWriter(open(flags.train_out, 'w'), ["id", "question", "correctAnswer"] +
                           answer_choices)
    train_out.writeheader()

    test_out = DictWriter(open(flags.test_out, 'w'), ["id", "question"] + answer_choices)
    test_out.writeheader()

    key_out = DictWriter(open(flags.key_out, 'w'), ["id", "correctAnswer"])
    key_out.writeheader()

    # Now write the questions out
    for qq in questions.values():
        print(qq.fold)
        if qq.fold == "devtest":
            test_out.writerow(qq.csv_line(kCHOICEIDS, "test"))
            key_out.writerow(qq.csv_line(kCHOICEIDS, "key"))
        else:
            train_out.writerow(qq.csv_line(kCHOICEIDS, "train"))
