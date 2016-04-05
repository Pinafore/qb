import argparse
from collections import defaultdict
from csv import DictWriter, DictReader
import operator

from unidecode import unidecode

from util.qdb import QuestionDatabase
from extract_expo_features import kEXPO_START, add_expo_questions

kBUZZ_OUT = ["question", "sentence", "word", "page", "evidence", "final", "weight"]
kPERF_OUT = ["question", "sentence", "token", "guess", "answer", "corr", "weight",
             "present_forward", "present_backward", "vw"]
kQUES_OUT = ["id", "answer", "sent", "text"]


class PositionIterator:
    """
    Given metadata and predictions, return a dictionary for each guess
    """

    def __init__(self, metadata, predictions, answers):
        self._meta = metadata
        self._pred = predictions

        self._answers = answers

    def add_expo_answers(self, expo_file):
        """
        The database doesn't have the right answer, so add answer from the expo
        file so we have them.
        """

        current_id = kEXPO_START
        with open(expo_file) as infile:
            id = kEXPO_START
            for ii in DictReader(infile):
                current_id += 1
                self._answers[current_id] = ii["answer"]

    def __iter__(self):
        for meta, prediction in zip(self._meta, self._pred):
            question, sent, token, guess = meta.split("\t")
            question = int(question)
            sent = int(sent)
            token = int(token)
            guess = guess.strip()

            pred_split = prediction.split()
            score = float(pred_split[0])
            name = pred_split[1]
            name_q, name_s, name_t = name.split("_")
            assert int(name_q) == question
            assert int(name_s) == sent, "%s vs %s" % (mm, name)
            assert int(name_t) == token

            buffer = defaultdict(dict)

            # Save the score and whether it was correct
            right_answer = (guess == self._answers[question])
            buffer[(sent, token)][guess] = (score, right_answer)
            yield question, buffer


def answer_presence(all_guesses):
    presence = set()
    for ss, tt in sorted(all_guesses):
        guesses = all_guesses[(ss, tt)]
        if any(y for x, y in guesses.values()):
            presence.add((ss, tt))
    return presence


def top_guesses(guesses):
    """
    Look at scores.  If any are above zero, that's a buzz.  If there are
    multiple above zero, take the highest.

    Returns a tuple of: what you would guess, whether you buzz, the associated
    score, and whether the correct answer was in the set
    """

    high_keys = sorted(guesses.items(), reverse=True,
                       key=operator.itemgetter(1))[:5]
    for guess, score in high_keys:
        if guess == high_keys[0][0] and guesses[guess][0] > 0:
            # We are buzzing
            yield guess, 1, score[0]
        else:
            # We are not buzzing
            yield guess, 0, score[0]


def final_guess(positions):
    """
    Go to the end of the question and gives the best answer at the end
    """

    last_position = max(positions)
    high_score = max(positions[last_position].values())
    high_keys = [x for x in positions[last_position] if positions[last_position][x] == high_score]

    return high_keys[0]


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--buzzes', type=str, default='',
                        help='Where we write resulting buzzes')
    parser.add_argument('--perf', type=str, default='',
                        help='Where we write performance statistics')
    parser.add_argument('--pred', type=str, default='',
                        help='Where we read predictions')
    parser.add_argument('--meta', type=str, default='',
                        help='Where we read metadata values')
    parser.add_argument('--qbdb', type=str, default='data/questions.db',
                        help="Source of questions")
    parser.add_argument('--vw_config', type=str, default='',
                        help="Configuration of classifier")
    parser.add_argument('--neg_weight', type=float, default=0.0,
                        help="Negative example weight")
    parser.add_argument('--question_out', type=str, default='',
                        help="Where we write out questions for buzzer")
    parser.add_argument('--finals', type=str, default='',
                        help="Where we write out answer after entire question")
    parser.add_argument('--expo', type=str, default='',
                        help="The expo file")

    flags = parser.parse_args()
    qdb = QuestionDatabase(flags.qbdb)

    buzz = DictWriter(open(flags.buzzes, 'w'), fieldnames=kBUZZ_OUT)
    buzz.writeheader()

    final_out = DictWriter(open(flags.finals, 'w'), fieldnames=["question", "answer"])
    final_out.writeheader()

    # Check file length
    with open(flags.meta) as infile:
        meta_lines = sum(1 for line in infile)
    with open(flags.pred) as infile:
        pred_lines = sum(1 for line in infile)
    assert meta_lines == pred_lines, "Prediction and meta files mismatch" + \
                                     "(%s: %i vs %s: %i)" % (
                                     flags.meta, meta_lines, flags.pred, pred_lines)

    pi = PositionIterator(open(flags.meta), open(flags.pred), qdb.all_answers())

    if flags.expo:
        pi.add_expo_answers(flags.expo)

    questions_with_buzzes = defaultdict(dict)
    for question_id, positions in pi:
        any_buzz = False
        final_sent, final_char = max(positions)
        presence = answer_presence(positions)
        for ss, tt in sorted(positions):
            for guess, final, weight in top_guesses(positions[(ss, tt)]):
                if (ss == final_sent and tt == final_char and not any_buzz) or final == 1:
                    any_buzz = True

                    # Add the question so it will be written out in the
                    # performance file
                    questions_with_buzzes[question_id]["question"] = question_id
                    questions_with_buzzes[question_id]["guess"] = guess.strip()
                    questions_with_buzzes[question_id]["sentence"] = ss
                    questions_with_buzzes[question_id]["token"] = tt
                    if any(x >= ss for x, y in presence):
                        questions_with_buzzes[question_id]["present_forward"] \
                            = min(x - ss for x, y in presence if x >= ss)
                    else:
                        questions_with_buzzes[question_id]["present_forward"] = -1

                    if any(x <= ss for x, y in presence):
                        questions_with_buzzes[question_id]["present_backward"] \
                            = min(ss - x for x, y in presence if x <= ss)
                    else:
                        questions_with_buzzes[question_id]["present_backward"] = -1

                d = {"question": question_id, "sentence": ss, "word": tt,
                     "page": guess.strip(), "evidence": "", "final": final,
                     "weight": weight}
                buzz.writerow(d)
            if any_buzz:
                break

        final_answer = final_guess(positions)
        d = {"question": question_id, "answer": final_answer}
        final_out.writerow(d)

    # Write out the questions

    if flags.expo:
        questions = add_expo_questions(flags.expo)
    else:
        questions = qdb.questions_with_pages()

    if flags.question_out:
        question_out = DictWriter(open(flags.question_out, 'w'), kQUES_OUT)
        question_out.writeheader()
    perf_out = DictWriter(open(flags.perf, 'w'), fieldnames=kPERF_OUT)
    perf_out.writeheader()
    for pp in questions:
        for qq in questions[pp]:
            if qq.qnum in questions_with_buzzes:
                # Write text for buzzer
                if flags.question_out:
                    for ll in qq.text_lines():
                        question_out.writerow(ll)

                # Write performance
                d = questions_with_buzzes[qq.qnum]
                d["corr"] = (d["guess"] == qq.page)
                d["answer"] = unidecode(qq.page)
                d["weight"] = flags.neg_weight
                d["vw"] = flags.vw_config
                perf_out.writerow(d)


if __name__ == "__main__":
    main()
