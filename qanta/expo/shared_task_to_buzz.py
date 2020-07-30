import nltk
import json
import argparse
from csv import DictWriter, DictReader
from random import shuffle
from string import split
from unidecode import unidecode

sent_detector = nltk.data.load("tokenizers/punkt/english.pickle")

# Old version had logic for powers, but removed
# https://raw.githubusercontent.com/Pinafore/qb/2567a885a356d488c3af7979140c09226b233b25/util/shared_task_to_buzz.py


def word_position_to_sent(questions, question, position):
    assert question in questions, "%i not in questions" % question
    count = 0
    for ss, sent in enumerate(questions[question]):
        for ww, word in enumerate(sent.split()):
            count += 1
            if count >= position:
                return ss, ww
    return ss, ww


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--shared_task", type=str, default="", help="Answers from shared task"
    )
    parser.add_argument(
        "--shared_task_offset",
        type=int,
        default=0,
        help="Starting ID for shared task input",
    )
    parser.add_argument(
        "--question_offset",
        type=int,
        default=0,
        help="Starting ID for questions file offset",
    )
    parser.add_argument("--questions", type=str, default="", help="Text of questions")
    parser.add_argument(
        "--sent_sep_ques",
        type=str,
        default="results/st/questions.csv",
        help="Sentence separated guesses",
    )
    parser.add_argument(
        "--final",
        type=str,
        default="results/st/final.csv",
        help="Where we write final answers",
    )
    parser.add_argument(
        "--buzz",
        type=str,
        default="results/st/buzz.csv",
        help="Where we write resulting buzzes",
    )

    flags = parser.parse_args()

    results = {}

    # Read in the questions so that we can convert absolute word
    # positions to sentence, word positions
    questions = {}
    answers = {}
    for ii in json.loads(open(flags.questions).read())["questions"]:
        question_id = int(ii["qid"]) + flags.question_offset
        questions[question_id] = sent_detector.tokenize(ii["question"])
        answers[question_id] = ii["answer"]
    print(questions)

    # TODO: If we care about more than one user (entrant), need to change logic here
    for ii in open(flags.shared_task):
        user, question, pos, guess = split(ii.strip(), maxsplit=3)
        question = int(question) + flags.shared_task_offset
        pos = int(pos)

        if question not in results or results[question][0] > pos:
            results[question] = (pos, guess)

    # Write out the questions, buzzes, and finals
    o_questions = DictWriter(
        open(flags.sent_sep_ques, "w"), ["id", "answer", "sent", "text"]
    )
    o_questions.writeheader()

    o_buzz = DictWriter(
        open(flags.buzz, "w"),
        ["question", "sentence", "word", "page", "evidence", "final", "weight"],
    )
    o_buzz.writeheader()

    o_final = DictWriter(open(flags.final, "w"), ["question", "answer"])
    o_final.writeheader()

    for question in results:
        pos, guess = results[question]
        ss, tt = word_position_to_sent(questions, question, pos)

        for sent_offset, sent in enumerate(questions[question]):
            question_line = {}
            question_line["id"] = question
            question_line["answer"] = unidecode(answers[question])
            question_line["sent"] = sent_offset
            question_line["text"] = unidecode(sent)
            o_questions.writerow(question_line)

        buzz_line = {}
        buzz_line["question"] = question
        buzz_line["sentence"] = ss
        buzz_line["word"] = tt
        buzz_line["page"] = guess
        buzz_line["final"] = 1
        buzz_line["weight"] = 1.0
        o_buzz.writerow(buzz_line)

        final_line = {}
        final_line["question"] = question
        final_line["answer"] = guess
        o_final.writerow(final_line)
