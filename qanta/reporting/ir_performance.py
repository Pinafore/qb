from qanta.extractors.ir import IrExtractor
from qanta.util.qdb import QuestionDatabase
from qanta.util.environment import QB_QUESTION_DB

from functional import pseq


def compute_stats():
    qdb = QuestionDatabase(QB_QUESTION_DB)
    ir = IrExtractor()
    questions = qdb.guess_questions()
    test_guesses = pseq(questions, partition_size=100)\
        .filter(lambda q: q.fold == 'test')\
        .map(lambda q: (q.page, ir.text_guess(q.flatten_text())))
    correct = 0
    close = 0
    total = 0
    for page, guesses in test_guesses:
        top_guess = max(guesses.items(), key=lambda x: x[1], default=None)
        if top_guess is not None and page == top_guess[0]:
            correct += 1
        elif page in guesses:
            close += 1
        total += 1
    print("Total Correct: {0}, Percent Correct: {1}".format(correct, correct / total))
    print("Total Close: {0}, Percent Close: {1}".format(close, close / total))
