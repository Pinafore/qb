# pylint: disable=too-many-locals
import sys
from collections import defaultdict
from qanta.util.qdb import QuestionDatabase
from qanta.util.constants import N_GUESSES


def process_file(filename):
    with open(filename, 'r') as f:
        questions = defaultdict(set)
        for line in f:
            tokens = line.split()
            offset = 1 if int(tokens[0]) == -1 else 0
            ident = tokens[1 + offset].replace("'", "").split('_')
            q = int(ident[0])
            s = int(ident[1])
            t = int(ident[2])
            guess = tokens[3 + offset]
            questions[(q, s, t)].add(guess)
        qdb = QuestionDatabase('data/questions.db')
        answers = qdb.all_answers()
        recall = 0
        warn = 0
        for ident, guesses in questions.items():
            if len(guesses) < N_GUESSES:
                print("WARNING LOW GUESSES")
                print('Question {0} is missing guesses, only has {1}'.format(ident, len(guesses)))
                warn += 1
            correct = answers[ident[0]].replace(' ', '_') in guesses
            recall += correct
        print('Recall: {0} Total: {1}'.format(recall / len(questions), len(questions)))
        print('Warned lines: {0}'.format(warn))


if __name__ == '__main__':
    process_file(sys.argv[1])
