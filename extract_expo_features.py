import argparse
from csv import DictReader, DictWriter
from collections import defaultdict

from nltk import sent_tokenize

from unidecode import unidecode

from qanta.extractors.lm import *
from qanta.util.qdb import QuestionDatabase, Question
from qanta.util.constants import FEATURES
from util.guess import GuessList
from extract_features import (
    instantiate_feature,
    feature_lines,
    guesses_for_question
)

kEXPO_START = 700000000
kQUES_OUT = ["id", "answer", "sent", "text"]


def add_expo_questions(exp_file, all_questions=defaultdict(set),
                       id_start=kEXPO_START):
    current_id = id_start
    num_questions = 0
    with open(exp_file) as infile:
        lines = DictReader(infile)
        for line in lines:
            num_questions += 1
            current_id += 1
            ans = unidecode(line["answer"].replace("_", " "))
            ques = Question(current_id, ans,
                            "UNK", True, None,
                            ans,
                            "UNK", "expo", None)
            for snum, sent in enumerate(sent_tokenize(unidecode(line["text"]))):
                ques.add_text(snum, sent)
            all_questions[ans].add(ques)
    print("Loaded %i questions from %s" % (num_questions, exp_file))
    return all_questions


def write_question_text(questions, output):
    with open(output, 'w') as outfile:
        question_out = DictWriter(outfile, kQUES_OUT)
        question_out.writeheader()
        for ii in questions:
            print("Writing out %s" % str(ii))
            for qq in questions[ii]:
                for ll in qq.text_lines():
                    question_out.writerow(ll)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--gap', type=int, default=4,
                        help='Gap (in number of tokens) between each guess')
    parser.add_argument('--guess_db', type=str, default='data/expo_guess.db',
                        help='Where we write/read the guesses')
    parser.add_argument('--question_db', type=str, default='data/questions.db')
    parser.add_argument("--granularity", type=str,
                        default="word")
    parser.add_argument("--expo", type=str,
                        default="data/expo.csv",
                        help="Where we read expo file")
    parser.add_argument("--question_out", type=str,
                        default='results/expo/questions.csv',
                        help="Where we write out questions")

    flags = parser.parse_args()
    # Load in the exposition questions
    questions = add_expo_questions(flags.expo)
    write_question_text(questions, flags.question_out)

    # Create question database
    qdb = QuestionDatabase(flags.question_db)

    # Create database for guess list
    guess_list = GuessList(flags.guess_db)

    # Generate all of the guess and store them in a guess_list
    features_that_guess = {"deep": instantiate_feature("deep", qdb)}

    for page in questions:
        for qq in questions[page]:
            guesses = guesses_for_question(qq, features_that_guess,
                                           guess_list, flags.gap)
            print(guesses)

            for guesser in guesses:
                guess_list.add_guesses(guesser, qq.qnum, "expo",
                                       guesses[guesser])
    del features_that_guess

    # Generate the features serially
    # for ff in ["label", "wikilinks"]:
    for ff in ["label"] + list(FEATURES.keys()):
        print("Loading %s" % ff)
        feat = instantiate_feature(ff, qdb)
        if ff == "label":
            meta = open("features/expo/%s.meta" % flags.granularity, 'w')
        else:
            meta = None

        # Open the feature file for output
        filename = ("features/%s/%s.%s.feat" %
                    ('expo', flags.granularity, ff))
        print("Opening %s for output" % filename)
        o = open(filename, 'w')

        line_num = 0
        for page in questions:
            for qq in questions[page]:
                for ss, tt, pp, line in feature_lines(qq, guess_list,
                                                      flags.granularity, feat):
                    line_num += 1
                    assert ff is not None
                    o.write("%s\n" % line)

                    if not meta is None:
                        meta.write("%i\t%i\t%i\t%s\n" %
                                    (qq.qnum, ss, tt, unidecode(pp)))

                    if line_num % 10000 == 0:
                        print("%s %s %s" % (page, pp, line))

                o.flush()
        o.close()
        print("Done with %s" % ff)
        # now that we're done with it, delete the feature
        del feat
