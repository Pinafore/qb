from __future__ import print_function
import textwrap
from collections import defaultdict
from glob import glob
import argparse
import random
from csv import DictReader, DictWriter
from time import sleep
import datetime

# from str import lower
from random import shuffle
import sys
import os
import json
import pdb
import csv
import pandas as pd
import inflect
import ast

#pip install torch==2.4.1
#pip install qa-metrics==0.2.17
#pip install inflect

from qa_metrics.pedant import PEDANT
from qa_metrics.transformerMatcher import TransformerMatcher
from qa_metrics.em import em_match
from qa_metrics.transformerMatcher import TransformerMatcher
tm = TransformerMatcher("zli12321/answer_equivalence_tiny_bert")
pedant = PEDANT()
p = inflect.engine()

kSHOW_RIGHT = False
kPAUSE = 0.25

kBIGNUMBERS = {
    -1: """








88888888
88888888





""",
    0: """

    .n~~%x.
  x88X   888.
 X888X   8888L
X8888X   88888
88888X   88888X
88888X   88888X
88888X   88888f
48888X   88888
 ?888X   8888"
  "88X   88*`
    ^"==="`



""",
    1: """

      oe
    .@88
==*88888
   88888
   88888
   88888
   88888
   88888
   88888
   88888
'**%%%%%%**



""",
    2: """

  .--~*teu.
 dF     988Nx
d888b   `8888>
?8888>  98888F
 "**"  x88888~
      d8888*`
    z8**"`   :
  :?.....  ..F
 <""888888888~
 8:  "888888*
 ""    "**"`



""",
    3: """

  .x~~"*Weu.
 d8Nu.  9888c
 88888  98888
 "***"  9888%
      ..@8*"
   ````"8Weu
  ..    ?8888L
:@88N   '8888N
*8888~  '8888F
'*8"`   9888%
  `~===*%"`



""",
    4: """

        xeee
       d888R
      d8888R
     @ 8888R
   .P  8888R
  :F   8888R
 x"    8888R
d8eeeee88888eer
       8888R
       8888R
    "*%%%%%%**~



""",
    5: """

  cuuu....uK
  888888888
  8*888**"
  >  .....
  Lz"  ^888Nu
  F     '8888k
  ..     88888>
 @888L   88888
'8888F   8888F
 %8F"   d888"
  ^"===*%"`



""",
    6: """

    .ue~~%u.
  .d88   z88i
 x888E  *8888
:8888E   ^""
98888E.=tWc.
98888N  '888N
98888E   8888E
'8888E   8888E
 ?888E   8888"
  "88&   888"
    ""==*""



""",
    7: """

dL ud8Nu  :8c
8Fd888888L %8
4N88888888cuR
4F   ^""%""d
d       .z8
^     z888
    d8888'
   888888
  :888888
   888888
   '%**%



""",
    8: """

   u+=~~~+u.
 z8F      `8N.
d88L       98E
98888bu.. .@*
"88888888NNu.
 "*8888888888i
 .zf""*8888888L
d8F      ^%888E
88>        `88~
'%N.       d*"
   ^"====="`



""",
    9: """

  .xn!~%x.
 x888   888.
X8888   8888:
88888   X8888
88888   88888>
`8888  :88888X
  `"**~ 88888>
 .xx.   88888
'8888>  8888~
 888"  :88%
  ^"===""


""",
}


class Score:
    def __init__(self, even=0, odd=0, human=0, computer=0):
        self.even = even
        self.odd = odd
        self.human = human
        self.computer = computer

    def add(self, score):
        return Score(
            self.even + score.even,
            self.odd + score.odd,
            self.human + score.human,
            self.computer + score.computer,
        )


class kCOLORS:
    PURPLE = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    @staticmethod
    def print(text, color="RED", end="\n"):
        start = getattr(kCOLORS, color)
        print(start + text + kCOLORS.ENDC, end=end)


def parse_final(final_string):
    """
    We have a string to show if we're at the end of the question, this code makes it more robust.
    """
    if final_string.lower() == "false":
        return 0
    if final_string.lower() == "true":
        return 1
    else:
        return int(final_string)


def write_readable(filename, ids, questions, buzzes, question_equivalents):
    question_num = 0
    o = open(filename, "w")
    # For each question
    for ii in ids:
        correct = [questions.answer(ii)] + question_equivalents[questions.answer(ii)]
        full_question_text = ' '.join(questions[ii].values())
        question_num += 1
        o.write("%i) " % question_num)
        power_found = False
        # For each sentence in the question
        model_buzz_bool = False
        for jj in questions[ii]:
            if (
                questions._power(ii)
                and not power_found
                and questions._power(ii).lower() in questions[ii][jj].lower()
            ):
                power_found = True
                o.write(
                    "%s  " % questions[ii][jj].replace(power(ii), "(*) %s" % power(ii))
                )
            else:
                question_id = ii
                ss = jj
                words = questions[ii][ss].split()
                new_words = []
                for wii, ww in enumerate(words):
                    current_guesses = buzzes.current_guesses(question_id, ss, wii - 1)
                    buzz_now = [x for x in current_guesses.values() if x.final]
                    if len(buzz_now) > 0:
                        model_guess = buzz_now[0].page
                        if not model_buzz_bool:
                            # Add the model buzz annotations
                            correctness = "+" if questions.answer_check(correct, model_guess, full_question_text) else "-"
                            new_words.append(f'({correctness})')
                            model_buzz_bool = True
                    if wii > 0:
                        new_words.append(words[wii - 1])
                # Add the last word
                new_words.append(ww)
                question_w_ann = ' '.join(new_words)
                o.write("%s  " % question_w_ann)
        model_final_correctness = '+' if questions.answer_check(correct, model_guess, full_question_text) else '-'
        o.write("\nMODEL FINAL GUESS: %s (%s)" % (model_guess, model_final_correctness))
        o.write("\nANSWER: %s\n\n" % correct)


def clear_screen():
    # print("Clearing")
    os.system("cls" if os.name == "nt" else "clear") # TEMP


class PowerPositions:
    def __init__(self, filename):
        self._power_marks = {}
        if filename:
            try:
                infile = DictReader(open(filename, "r"))
                for ii in infile:
                    question = int(ii["question"])
                    self._power_marks[question] = ii["word"]
            except:
                print("Couldn't load from %s" % filename)
            print(
                "Read power marks from %s: %s ..."
                % (filename, str(self._power_marks.keys())[1:69])
            )
        else:
            print("No power marks")

    def __call__(self, question):
        if question in self._power_marks:
            return self._power_marks[question]
        else:
            return ""


# Utilities for single character input
class _Getch:
    """Gets a single character from standard input.  Does not echo to the
screen."""

    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self):
        return self.impl()


class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
            sys.stdin.flush()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt

        return msvcrt.getch()


getch = _Getch()


def show_score(
    left_score,
    right_score,
    left_header="HUMAN",
    right_header="COMPUTER",
    left_color="GREEN",
    right_color="BLUE",
    flush=True,
):
    if flush:
        clear_screen()
    # Print the header
    print("%-15s" % "", end="")
    kCOLORS.print("%-15s" % left_header, left_color, end="")
    print("%-30s" % "", end="")
    kCOLORS.print("%-15s\n" % right_header, right_color)

    for line in range(1, 15):
        for num, color in [(left_score, left_color), (right_score, right_color)]:
            for place in [100, 10, 1]:
                if place == 100 and num < 0:
                    val = -1
                else:
                    val = (abs(num) % (place * 10)) // place

                kCOLORS.print(
                    "%-15s" % kBIGNUMBERS[val].split("\n")[line], color=color, end=" "
                )
            print("|", end=" ")
        print(" ")


class Guess:
    def __init__(self, system, page, evidence, final, weight):
        self.system = system
        self.page = page.replace("_", " ")
        self.evidence = evidence
        self.final = final
        self.weight = weight


class Buzzes:
    def __init__(self, file_path, questions):
        self._buzzes = defaultdict(dict)
        self._finals = defaultdict(dict)

        self._questions = questions
        print("Initializing buzz files")

    def debug(self):
        self.add_guess(0, 0, 5, "A", "Heisenberg", "", 0, 0.2)
        self.add_guess(0, 0, 5, "C", "Narcos", "", 0, 0.2)
        self.add_guess(0, 2, 3, "A", "Better_Call_Saul", "", 1, 0.7)
        self.add_guess(0, 2, 3, "B", "Breaking Bad", "", 1, 0.6)
        self.add_guess(0, 2, 3, "C", "Breaking Bad", "", 1, 0.5)
        self._finals[0]["A"] = "Breaking Bad"
        self._finals[0]["B"] = "Breaking Bad"
        self._finals[0]["C"] = "Breaking Bad"

        self.add_guess(1, 0, 5, "A", "SimCity", "", 0, 0.2)
        self.add_guess(1, 0, 5, "C", "Skyrim", "", 0, 0.2)
        self.add_guess(1, 2, 3, "A", "Skyrim", "", 0, 0.7)
        self.add_guess(1, 3, 3, "B", "Jedi Knight", "", 0, 0.6)
        self.add_guess(1, 3, 3, "C", "Jedi Knight", "", 0, 0.5)
        self._finals[1]["A"] = "Fallout 76"
        self._finals[1]["B"] = "Fallout 76"
        self._finals[1]["C"] = "Fallout (series)"

        self.add_guess(2, 0, 5, "A", "Apple", "", 0, 0.2)
        self.add_guess(2, 0, 5, "C", "Onion", "", 0, 0.2)
        self.add_guess(2, 0, 5, "C", "Apple", "", 0, 0.2)
        self.add_guess(2, 2, 3, "A", "Apple", "", 1, 0.7)
        self.add_guess(2, 3, 3, "B", "Onion", "", 0, 0.6)
        self.add_guess(2, 3, 3, "C", "Jedi Knight", "", 0, 0.5)
        self._finals[2]["A"] = "Potato"
        self._finals[2]["B"] = "Potato"
        self._finals[2]["C"] = "Potato"

    def add_guess(self, question, sent, word, system, guess, evidence, final, weight):
        if not (sent, word) in self._buzzes[question]:
            self._buzzes[question][(sent, word)] = {}
        if final != 0 and sent == 0 and word < 25:
            final = 0
        self._buzzes[question][(sent, word)][guess] = Guess(
            system, guess, evidence, final, weight
        )

    def add_system(self, file_path):
        buzzfile = DictReader(open("%s.buzz.csv" % file_path, "r"))
        system = file_path.replace("CMSC723_", "").split("/")[-1]
        #system = system.split(".")[0]
        #system = system.split("_")[0]
        system = system.replace("_"," ")

        for ii in buzzfile:
            question, sent, word = (
                int(ii["question"]),
                int(ii["sentence"]),
                int(ii["word"]),
            )
            self.add_guess(
                question,
                sent,
                word,
                system,
                ii["page"],
                ii["evidence"],
                parse_final(ii["final"]),
                float(ii["weight"]),
            )

        self.load_finals(system, "%s.final.csv" % file_path)

    def load_finals(self, system, final_file):
        ff = DictReader(open(final_file))
        for ii in ff:
            self._finals[int(ii["question"])][system] = ii["answer"].replace("_", " ")

    def current_guesses(self, question, sent, word):
        try:
            ss, ww = max(
                x
                for x in self._buzzes[question]
                if x[0] < sent or (x[0] == sent and x[1] <= max(0, word))
            )

        except ValueError:
            return {}

        assert (ss, ww) in self._buzzes[question]
        return self._buzzes[question][(ss, ww)]

    def __iter__(self):
        for ii in self._buzzes:
            yield ii


class Questions:
    def __init__(self):
        self._questions = defaultdict(dict)
        self._answers = defaultdict(str)
        self._power = PowerPositions("")
        self._equivalents = {}

        print("Initializing questions")

    def answer_check(self, reference, guess, question):
        def metric_em_match(reference_answer, candidate_answer):
            match_result = em_match(reference_answer, candidate_answer)
            return match_result

        def metric_pedant(reference_answer, candidate_answer, question):
            match_result = pedant.evaluate(reference_answer, candidate_answer, question)
            return match_result

        def metric_pedant_scores(reference_answer, candidate_answer, question):
            match_result = pedant.get_scores(reference_answer, candidate_answer, question)
            return match_result

        def metric_neural(reference_answer, candidate_answer):
            # Supported models: zli12321/answer_equivalence_roberta-large, zli12321/answer_equivalence_tiny_bert, zli12321/answer_equivalence_roberta, zli12321/answer_equivalence_bert, zli12321/answer_equivalence_distilbert, zli12321/answer_equivalence_distilroberta
            #scores = tm.transformer_match(reference_answer, candidate_answer, question)
            match_result = tm.transformer_match(reference_answer, candidate_answer, question)
            return match_result

        def metric_neural(reference_answer, candidate_answer, question):
            # Supported models: zli12321/answer_equivalence_roberta-large, zli12321/answer_equivalence_tiny_bert, zli12321/answer_equivalence_roberta, zli12321/answer_equivalence_bert, zli12321/answer_equivalence_distilbert, zli12321/answer_equivalence_distilroberta
            #scores = tm.transformer_match(reference_answer, candidate_answer, question)

            match_result = tm.transformer_match(reference_answer, candidate_answer, question)
            return match_result

        # def answer_equali(packet1, gold1, question):
        #     packet1['reference_answer'] = packet1.apply(lambda row: gold1.iloc[row['question_index']-1]['reference'],axis=1)
        #     packet1['em_match'] = packet1.apply(lambda row: metric_em_match(row['reference_answer'], row['prediction']),axis=1)
        #     packet1['pendant_evaluate'] = packet1.apply(lambda row: metric_pedant(row['reference_answer'], row['prediction'], row['question']),axis=1)
        #     packet1['pedant_neural'] = packet1.apply(lambda row: metric_neural(row['reference_answer'], row['prediction'], row['question']),axis=1)
        #     return packet1

        def normalize_apostrophe(text):
            return text.replace("’", "'")

        def preprocess(text):
            text = normalize_apostrophe(text.strip()).lower()
            return text

        def doublecheck_plural(reference_answers, answer1):
            answer_equal_list = []
            for ref in reference_answers:
                answer2 = ref
                if p.singular_noun(answer1) == answer2 or p.singular_noun(answer2) == answer1:
                    answer_equal_list.append(True)
                else:
                    answer_equal_list.append(False)
            return any(answer_equal_list)

        ref_p = [preprocess(item) for item in reference]
        if guess != None:
            guess_p = preprocess(guess)
        else:
            guess_p = None
        qanta_pedant_neural = metric_neural(ref_p, guess_p, question)
        qanta_double_check = doublecheck_plural(ref_p, guess_p)
        if (qanta_pedant_neural==False) and (qanta_double_check==True):
            result =  True
        else:
            result = qanta_pedant_neural

        return result

    def debug(self):
        self._questions[0] = {
            0: "His aliases include: Pastor Hansford of Free Will Baptist in Coushatta, Louisianna; Viktor with a K St. Claire, a South African who just inherited money from his uncle; Charlie Hustle, a mailboy in the HHM mailroom; and Gene Takavic, a Cinnabon manager in Omaha.",
            1: "His best known alias came from selling burner phones in Albuquerque, New Mexico and was more fitting for a lawyer than his birthname, James McGill.",
            2: "For ten points, name this titular CRIMINAL lawyer from an AMC series who originated on Breaking Bad.",
        }
        self._answers[0] = "Better Call Saul"

        self._questions[1] = {
            0: 'This game contains an easter egg based on fan art of blocks spelling out "P A M" and an insect in a top hat, and its launch was declared a holiday by Jim Justice on November 14, 2018.',
            1: 'The "Personal Matters" quest requires the player to kill Evan, who was the fiancé of the Overseer.  Locations in this game include the Whitespring Resort, which is based on the real-life Greenbrier resort in Solphur Springs, West Virginia.',
            2: "A multiplayer game centered on a control vault, for ten points name Bathesda's most recent entry in the Fallout franchise.",
        }
        self._answers[1] = "Fallout 76"

        self._questions[2] = {
            0: "Amsterdam experienced a riot named for this crop during World War One.",
            1: "In the 1830s and 40s, Russian peasants on the lower Volga objected to the forced introduction of this crop in a namesake series of riots.",
            2: "Antoine-Augustin Parmentier pushed for the cultivation of this crop in France.",
            3: "This crop provides an alternate name for the War of the Bavarian Succession.",
            4: "An epidemic of phytophthora infestans devastated the cultivation of this crop in the mid-19th Century.",
            5: "Robert Peel repealed the Corn Laws to counteract a blight of, for 10 points, what crop that caused an Irish famine?",
        }
        self._answers[2] = "Potato"

    def load_equivalents(self, equivalent_file):
        if equivalent_file:
            with open(equivalent_file, 'r') as infile:
                self.equivalents = json.loads(infile.read())

            normalized = []
            for orig, replace in [(" ", "_"), ("_", " ")]:
                for title in self.equivalents:
                    if orig in title:
                        normalized.append((title, title.replace(orig, replace)))

            for orig, replace in normalized:
                self.equivalents[replace] = self.equivalents[orig]
            print(self.equivalents)

    def load_power(self, power_file):
        self._power = PowerPositions(power_file)

    def load_questions(self, question_file):
        qfile = DictReader(open(question_file, "r"))

        for ii in qfile:
            self._questions[int(ii["id"])][int(ii["sent"])] = ii["text"]
            self._answers[int(ii["id"])] = ii["answer"].strip().replace("_", " ")
        # print('In load_questions')
        # print(self._questions)
        # print(self._answers)

    def __iter__(self):
        for ii in self._questions:
            yield ii

    def __getitem__(self, val):
        return self._questions[val]

    def answer(self, val):
        return self._answers[val]


def format_display(
    display_num,
    question_text,
    sent,
    word,
    current_guesses,
    answer=None,
    guess_limit=5,
    points=10
):
    sep = "".join(["-"] * 80)

    current_text = ""
    for ss in range(sent):
        current_text += "%s " % question_text[ss]
    current_text += " ".join(question_text[sent].split()[:word])
    current_text = "\n".join(textwrap.wrap(current_text, 80))

    report = "Question %i: %i points\n%s\n%s\n%s\n\n" % (
        display_num,
        points,
        sep,
        current_text,
        sep,
    )

    for gg in sorted(
        current_guesses, key=lambda x: current_guesses[x].weight, reverse=True
    )[:guess_limit]:
        guess = current_guesses[gg]
        question_text_join = ' '.join(question_text.values())

        if questions.answer_check(answer, guess.page, question_text_join):
            report += "%-18s\t%-50s\t%0.2f\t%s\n" % (
                guess.system,
                "***CORRECT***",
                guess.weight,
                guess.evidence[:60],
            )
        else:
            report += "%-18s\t%-50s\t%0.2f\t%s\n" % (
                guess.system,
                guess.page,
                guess.weight,
                guess.evidence[:60],
            )
    return report


def interpret_keypress(other_allowable=""):
    """
    See whether a number was pressed (give terminal bell if so) and return
    value.  Otherwise returns none.  Tries to handle arrows as a single
    press.
    """
    press = getch()
    if press == "\x1b":
        getch()
        getch()
        press = "direction"

    if press.upper() in other_allowable:
        return press.upper()

    if press != "direction" and press != " ":
        try:
            press = int(press)
        except ValueError:
            press = None
    return press


def answer(ans, system):
    if system:
        print("%s says:" % system)
    os.system("afplay /System/Library/Sounds/Glass.aiff")
    os.system("say -v Tom %s" % ans.replace("'", "").split("(")[0])
    sleep(kPAUSE)
    #print(ans)


def setup_gameplay_writer(out_file):
    if out_file.endswith(".csv"):
        out_writer = DictWriter(open(out_file, 'w'), {"qid", "run_id", "sentence", "model_buzz", "model_guess", "model_correctness", "human_buzz", "human_correctness"})
        out_writer.writeheader()

        model_out_file = out_file.replace(".csv", " (model).csv")
        model_out_writer = DictWriter(open(model_out_file, 'w'), {"qid", "run_id", "sentence", "model_buzz", "model_guess", "model_correctness"})
        model_out_writer.writeheader()

        human_out_file = out_file.replace(".csv", " (human).csv")
        human_out_writer = DictWriter(open(human_out_file, 'w'), {"qid", "run_id", "sentence", "human_buzz", "human_correctness"})
        human_out_writer.writeheader()

        return {"out_writer": out_writer, "model_out_writer": model_out_writer, "human_out_writer": human_out_writer}
    else:
        return {"out_writer": None, "model_out_writer": None, "human_out_writer": None}


def write_gameplay_log(out_writer_dict, qid, run_id, run_text, model_buzz, model_guess, model_correctness, human_buzz, human_correctness):
    if out_writer_dict['out_writer']:
        out_writer_dict['out_writer'].writerow({
            "qid": qid,
            "run_id": run_id,
            "sentence": run_text,
            "model_buzz": model_buzz,
            "model_guess": model_guess,
            "model_correctness": model_correctness,
            "human_buzz": human_buzz,
            "human_correctness": human_correctness
        })
        if human_buzz == 'N/A':
            out_writer_dict['model_out_writer'].writerow({
                "qid": qid,
                "run_id": run_id,
                "sentence": run_text,
                "model_buzz": model_buzz,
                "model_guess": model_guess,
                "model_correctness": model_correctness
            })
        elif model_buzz == 'N/A':
            out_writer_dict['human_out_writer'].writerow({
                "qid": qid,
                "run_id": run_id,
                "sentence": run_text,
                "human_buzz": human_buzz,
                "human_correctness": human_correctness
            })


def present_question_hc(
    display_num,
    question_id,
    question_text,
    buzzes,
    final,
    correct,
    out_writer_dict,
    score=Score(),
    power="10"
):
    """
    Shows one question to a human and computer
    """

    human_delta = 0
    computer_delta = 0
    question_value = 15
    for ss in question_text:
        words = question_text[ss].split()
        for ii, ww in enumerate(words):
            # if we've reached the end of the question
            if ss == max(question_text) and ii == len(question_text[ss].split()) - 1:
                # If computer hasn't buzzed, let the computer buzz
                if computer_delta == 0:
                    print(final)
                    system = random.choice(list(final.keys()))
                    answer(final[system].split("(")[0], system)
                    final = final[system]
                    question_text_join = ' '.join(question_text.values())
                    answer_check = questions.answer_check(correct, final, question_text_join)
                    write_gameplay_log(out_writer_dict, question_id, ss, question_text[ss], ' '.join(words[:ii+1]), final, answer_check, 'N/A', 'N/A')
                    if answer_check:
                        return Score(human=human_delta, computer=10)
                    else:
                        print("Incorrect answer: %s" % final)
                else:
                    words += [" ", " ", " ", " ", " "]

            if ww.lower().startswith(power.lower()):
                question_value = 10
            press = interpret_keypress()
            current_guesses = buzzes.current_guesses(question_id, ss, ii - 1)
            buzz_now = [x for x in current_guesses.values() if x.final]

            #print(buzz_now)
            # Removing this assertion now that we can have multiple systems playing
            # assert len(buzz_now) < 2, "Cannot buzz on more than one thing"
            if isinstance(press, int):
                os.system("afplay /System/Library/Sounds/Glass.aiff")
                response = None
                while response is None:
                    response = input("Player %i, provide an answer:\t" % press)
                    if "+" in response:
                        write_gameplay_log(out_writer_dict, question_id, ss, question_text[ss], 'N/A', 'N/A', 'N/A', ' '.join(words[:ii+1]), True)
                        return Score(human=question_value, computer=computer_delta)
                    elif "-" in response:
                        write_gameplay_log(out_writer_dict, question_id, ss, question_text[ss], 'N/A', 'N/A', 'N/A', ' '.join(words[:ii+1]), False)
                        if computer_delta == -5:
                            # If computer already got it wrong, question is over
                            return Score(computer=computer_delta)
                        else:
                            human_delta = -5
                    else:
                        response = None
            # Don't buzz if anyone else has gotten it wrong
            elif buzz_now and human_delta == 0 and computer_delta == 0:
                show_score(
                    score.human + human_delta,
                    score.computer + computer_delta,
                    "HUMAN",
                    "COMPUTER",
                )
                print(
                    format_display(
                        display_num,
                        question_text,
                        ss,
                        ii + 1,
                        current_guesses,
                        answer=correct,
                        points=question_value
                    )
                )
                answer(buzz_now[0].page.split("(")[0], buzz_now[0].system)
                question_text_join = ' '.join(question_text.values())
                answer_check = questions.answer_check(correct, buzz_now[0].page, question_text_join)
                write_gameplay_log(out_writer_dict, question_id, ss, question_text[ss], ' '.join(words[:ii+1]), buzz_now[0].page, answer_check, 'N/A', 'N/A')
                if answer_check:
                    print("Computer guesses: %s (correct)" % buzz_now[0].page)
                    sleep(5)
                    return Score(human=human_delta, computer=question_value)
                else:
                    print("Computer guesses: %s (wrong)" % buzz_now[0].page)
                    sleep(5)
                    computer_delta = -5
                    show_score(
                        score.human + human_delta,
                        score.computer + computer_delta,
                        "HUMAN",
                        "COMPUTER",
                    )
                    format_display(
                        display_num,
                        question_text,
                        max(question_text),
                        0,
                        current_guesses,
                        answer=correct,
                        points=question_value
                    )
            else:
                show_score(
                    score.human + human_delta,
                    score.computer + computer_delta,
                    "HUMAN",
                    "COMPUTER",
                )
                print(
                    format_display(
                        display_num,
                        question_text,
                        ss,
                        ii + 1,
                        current_guesses,
                        answer=correct,
                        points=question_value,
                    )
                )

    if human_delta == 0:
        response = None
        while response is None:
            os.system("afplay /System/Library/Sounds/Glass.aiff")
            response = input("Player, take a guess:\t")
            if "+" in response:
                return Score(human=10, computer=computer_delta)
            elif "-" in response:
                return Score(computer=computer_delta)
            else:
                response = None

    return Score(human=human_delta, computer=computer_delta)


def create_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--questions", type=str, default="")
    parser.add_argument("--model_directory", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument(
        "--output",
        type=str,
        default="GAMEPLAY %s.csv"
        % datetime.datetime.now().strftime("%I:%M%p on %B %d %Y"),
        help="This parameter will only work if the output str ends with `.csv`.\
              Default is GAMEPLAY <time>.csv and three files will be generated (model, human, and combined behaviors).",
    )
    parser.add_argument("--players", type=int, default=1)
    parser.add_argument("--human_start", type=int, default=0)
    parser.add_argument("--computer_start", type=int, default=0)
    parser.add_argument("--odd_start", type=int, default=0)
    parser.add_argument("--even_start", type=int, default=0)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--power", type=str, default="")
    parser.add_argument("--max_questions", type=int, default=40)
    parser.add_argument("--answer_equivalents", type=str, default="")
    parser.add_argument("--readable", type=str, default="readable.txt",
                        help="The human-readable file of the questions with all pre-defined model buzzes.")
    return parser.parse_args()


def load_data(flags):
    questions = Questions()

    if flags.questions != "":
        questions.load_questions(flags.questions)
        questions.load_power(flags.power)
        questions.load_equivalents(flags.answer_equivalents)
    else:
        questions.debug()

    buzzes = Buzzes(flags.model_directory, questions)

    if flags.model_directory != "":
        for ii in glob("%s/*" % flags.model_directory):
            if ii.endswith(".buzz.csv"):
                # if we have a specific model, only load that
                if flags.model != "" and "%s.buzz.csv" % flags.model not in ii:
                    print("Not loading model %s" % ii)
                else:
                    print("Loading model %s" % ii)
                    buzzes.add_system(ii.replace(".buzz.csv", ""))
                    #print('Got Buzzes')
                    #print(buzzes)
    else:
        buzzes.debug()

    return questions, buzzes


def buzzer_check(players):
    if players > 0:
        print("Time for a buzzer check")
    players_needed = range(1, players + 1)
    current_players = set()
    while len(current_players) < len(players_needed):
        print(
            "Player %i, please buzz in"
            % min(x for x in players_needed if x not in current_players)
        )
        press = interpret_keypress()
        if press in players_needed:
            os.system("afplay /System/Library/Sounds/Glass.aiff")
            print("Thanks for buzzing in, player %i!" % press)
            current_players.add(press)

    if players > 0:
        sleep(1.5)
        answer("I'm ready too", "QANTA")


def check_hc_tie(score):
    """
    For the computer-human and human-human programs, this needs to be
    different.  This is why it's a silly function.
    """
    return score.human == score.computer


def question_loop(flags, questions, buzzes, present_question, check_tie):
    out_writer_dict = setup_gameplay_writer(flags.output)

    score = Score(
        odd=flags.odd_start,
        even=flags.even_start,
        human=flags.human_start,
        computer=flags.computer_start,
    )
    question_num = 0
    #question_ids = sorted(questions._questions.keys(), key=lambda x: x % 10)
    question_ids = questions._questions.keys()
    # print(question_ids)
    # print(list(buzzes))
    question_ids = [x for x in question_ids if x in buzzes]
    #print(question_ids)
    question_equivalents = questions.equivalents


    if flags.readable != "":
        write_readable(flags.readable, question_ids, questions, buzzes, question_equivalents)

    for ii in question_ids:
        question_num += 1
        if flags.skip > 0 and question_num < flags.skip:
            continue

        power_mark = questions._power(ii)
        if power_mark == "10":
            print(
                "Looking for power for %i, got %s %s"
                % (ii, power_mark, str(ii in power._power_marks.keys()))
            )

        score_delta = present_question(
            question_num,
            ii,
            questions[ii],
            buzzes,
            buzzes._finals[ii],
            [questions.answer(ii)] + question_equivalents[questions.answer(ii)],
            out_writer_dict=out_writer_dict,
            score=score,
            power=questions._power(ii)
        )
        score = score.add(score_delta)

        print(
            "Correct answer of Question %i: %s" % (question_num, questions.answer(ii))
        )
        sleep(kPAUSE)

        if question_num > flags.max_questions - 1:
            break

    if check_tie(score):
        print("Tiebreaker!")
        for ii in question_ids[question_num:]:
            question_num += 1
            score_delta = present_question(
                question_num,
                ii,
                questions[ii],
                buzzes,
                buzzes._finals[ii],
                questions.answer(ii),
                score=score,
                power=questions._power(ii),
            )
            score = score.add(score_delta)

            print(
                "Correct answer of Question %i: %s"
                % (question_num, questions.answer(ii))
            )
            sleep(kPAUSE)

    return score


if __name__ == "__main__":
    flags = create_parser()
    questions, buzzes = load_data(flags)
    print("Done loading data")

    clear_screen()
    buzzer_check(flags.players)

    score = question_loop(flags, questions, buzzes, present_question_hc, check_hc_tie)

    show_score(score.human, score.computer, "HUMAN", "COMPUTER")
