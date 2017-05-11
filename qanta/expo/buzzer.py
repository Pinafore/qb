import textwrap
from collections import defaultdict, Counter
import argparse
from csv import DictReader
from time import sleep
import os

from qanta.preprocess import format_guess
from qanta.datasets.quiz_bowl import QuizBowlDataset

kSHOW_RIGHT = False
kPAUSE = .25
kSYSTEM = "QANTA"

kBIGNUMBERS = {-1:
"""








88888888
88888888





""",
0:
"""

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
1:
"""

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
2:
"""

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
3:
"""

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
4:
"""

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
5:
"""

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
6:
"""

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
7:
"""

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
8:
"""

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
9:
"""

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


"""}


class kCOLORS:
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def print(text, color="RED", end='\n'):
        start = getattr(kCOLORS, color)
        print(start + text + kCOLORS.ENDC, end=end)


def write_readable(filename, questions, question_ids):
    question_num = 0
    o = open(filename, 'w')
    for qnum in question_ids:
        question_num += 1
        o.write("%i) " % question_num)
        for sent in questions[qnum]:
            o.write("%s  " % questions[qnum][sent])
        o.write("\nANSWER: %s\n\n" % questions.answer(qnum))
    o.close()


def clear_screen():
    print("Clearing")
    os.system('cls' if os.name == 'nt' else 'clear')


class PowerPositions:
    def __init__(self, filename):
        self._power_marks = {}
        try:
            infile = DictReader(open(filename, 'r'))
            for ii in infile:
                question = int(ii['question'])
                self._power_marks[question] = ii['word']
            print("Read power marks from %s: %s ..." %
                  (filename, str(self._power_marks.keys())[1:69]))
        except FileNotFoundError:
            pass

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

    def __call__(self): return self.impl()


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


def show_score(left_score, right_score,
               left_header="HUMAN", right_header="COMPUTER",
               left_color="GREEN", right_color="BLUE",
               flush=True):
    assert isinstance(left_score, int)
    assert isinstance(right_score, int)
    if flush:
        clear_screen()
    # Print the header
    print("%-15s" % "", end='')
    kCOLORS.print("%-15s" % left_header, left_color, end='')
    print("%-30s" % "", end='')
    kCOLORS.print("%-15s\n" % right_header, right_color)

    for line in range(1, 15):
        for num, color in [(left_score, left_color),
                           (right_score, right_color)]:
            for place in [100, 10, 1]:
                if place == 100 and num < 0:
                    val = -1
                else:
                    val = (abs(num) % (place * 10)) // place
                kCOLORS.print("%-15s" % kBIGNUMBERS[val].split("\n")[line],
                              color=color, end=' ')
            print("|", end=" ")
        print(" ")


class Guess:
    def __init__(self, page, evidence, final, weight):
        self.page = page
        self.evidence = evidence
        self.final = final
        self.weight = weight


class Buzzes:
    def __init__(self, buzz_file):
        buzzfile = DictReader(open(buzz_file, 'r'))

        self._buzzes = defaultdict(dict)
        for r in buzzfile:
            question, sent, word = int(r["question"]), int(r["sentence"]), int(r["word"])
            if not (sent, word) in self._buzzes[question]:
                self._buzzes[question][(sent, word)] = {}
            self._buzzes[question][(sent, word)][r["page"]] = \
                Guess(r["page"], r["evidence"], int(r["final"]), float(r["weight"]))

    def current_guesses(self, question, sent, word):
        try:
            ss, ww = max(x for x in self._buzzes[question] if
                            x[0] < sent or (x[0] == sent and x[1] <= max(0, word)))
        except ValueError:
            return {}

        assert (ss, ww) in self._buzzes[question]
        return self._buzzes[question][(ss, ww)]

    def __iter__(self):
        for ii in self._buzzes:
            yield ii

    def final_guess(self, question):
        for ss, ww in sorted(self._buzzes[question], reverse=True):
            for bb in self._buzzes[question][(ss, ww)]:
                if self._buzzes[question][(ss, ww)][bb].final:
                    return bb
        return None


class Questions:
    def __init__(self, question_file):
        qfile = DictReader(open(question_file, 'r'))

        self._questions = defaultdict(dict)
        self._answers = defaultdict(str)
        for r in qfile:
            self._questions[int(r["id"])][int(r["sent"])] = r["text"]
            self._answers[int(r["id"])] = r["answer"].strip()

    def __iter__(self):
        for qnum in self._questions:
            yield qnum

    def __getitem__(self, val):
        return self._questions[val]

    def answer(self, val):
        return self._answers[val]


def select_features(evidence_str, allowed_features):
    features = evidence_str.split()
    included_features = [f for f in features if f in allowed_features]
    return ' '.join(included_features)


def format_display(display_num, question_text, sent, word, current_guesses,
                   answer=None, guess_limit=5, points=10, disable_features=False, answerable=None):
    sep = "".join(["-"] * 80)

    current_text = ""
    for ss in range(sent):
        current_text += "%s " % question_text[ss]
    current_text += " ".join(question_text[sent].split()[:word])
    current_text = "\n".join(textwrap.wrap(current_text, 80))

    report = 'answerable: {}\n'.format(answerable)
    report += "Question %i: %i points\n%s\n%s\n%s\n\n" % \
        (display_num, points, sep, current_text, sep)

    top_guesses = sorted(current_guesses,
                         key=lambda x: current_guesses[x].weight, reverse=True)[:guess_limit]
    duplicated_feature_counter = Counter()
    for g in top_guesses:
        evidence = current_guesses[g].evidence.split()
        for f in evidence:
            duplicated_feature_counter[f] += 1

    allowed_features = set()
    for k, v in duplicated_feature_counter.items():
        if v == 1:
            allowed_features.add(k)

    if False and len(top_guesses) > 0:
        print(top_guesses)
        print(allowed_features)
        print(duplicated_feature_counter)
        raise Exception()
    for gg in top_guesses:
        guess = current_guesses[gg]
        if disable_features:
            features = ''
        else:
            features = select_features(guess.evidence, allowed_features)[:100]
        if guess.page == answer:
            report += "%s\t%f\t%s\n" % (
                "***CORRECT***",
                guess.weight,
                features
            )
        else:
            report += "%s\t%f\t%s\n" % (
                guess.page,
                guess.weight,
                features
            )
    return report


def load_finals(final_file):
    f = DictReader(open(final_file))
    d = {}
    for i in f:
        d[int(i['question'])] = i['answer']
    return d


def interpret_keypress():
    """
    See whether a number was pressed (give terminal bell if so) and return
    value.  Otherwise returns none.  Tries to handle arrows as a single
    press.
    """
    press = getch()
    if press == '\x1b':
        getch()
        getch()
        press = "direction"

    if press == 'Q':
        raise Exception('Exiting expo by user request from pressing Q')

    if press != "direction" and press != " ":
        try:
            press = int(press)
        except ValueError:
            press = None
    return press


def answer(ans, print_string="%s says:" % kSYSTEM):
    if print_string:
        print(print_string)
    os.system("afplay /System/Library/Sounds/Glass.aiff")
    os.system("say %s" % ans.replace("'", "").replace('_', '').split("(")[0])
    sleep(kPAUSE)
    print(ans)


def present_question(display_num, question_id, question_text, buzzes, final,
                     correct, human=0, computer=0, power="10", answerable=None):
    human_delta = 0
    computer_delta = 0
    question_value = 15
    for ss in sorted(question_text):
        words = question_text[ss].split()
        for ii, ww in enumerate(words):
            if str.lower(ww).startswith(str.lower(power)):
                question_value = 10
            press = interpret_keypress()
            current_guesses = buzzes.current_guesses(question_id, ss, ii - 2)
            buzz_now = [x for x in current_guesses.values() if x.final]
            assert len(buzz_now) < 2, "Cannot buzz on more than one thing"
            if isinstance(press, int):
                os.system("afplay /System/Library/Sounds/Glass.aiff")
                response = None
                while response is None:
                    response = input("Player %i, provide an answer:\t" % press)
                    if '+' in response:
                        return (human + question_value,
                                computer + computer_delta,
                                response[1:])
                    elif '-' in response:
                        if computer_delta == -5:
                            return human, computer + computer_delta, response[1:]
                        else:
                            human_delta = -5
                    else:
                        response = None
            # Don't buzz if anyone else has gotten it wrong
            elif buzz_now and human_delta == 0 and computer_delta == 0:
                show_score(human + human_delta,
                           computer + computer_delta,
                           "HUMAN", "COMPUTER")
                print(format_display(display_num, question_text, ss, ii + 1,
                                     current_guesses, answer=correct,
                                     points=question_value, answerable=answerable))
                answer(buzz_now[0].page)
                if buzz_now[0].page == correct:
                    print("Computer guesses: %s (correct)" % buzz_now[0].page)
                    sleep(1)
                    print(format_display(display_num, question_text, max(question_text), 0,
                                         current_guesses, answer=correct, points=question_value, answerable=answerable))
                    return (human + human_delta, computer + question_value,
                            buzz_now[0].page)
                else:
                    print("Computer guesses: %s (wrong)" % buzz_now[0].page)
                    sleep(1)
                    computer_delta = -5
                    show_score(human + human_delta,
                               computer + computer_delta,
                               "HUMAN", "COMPUTER")
                    print(format_display(display_num, question_text, max(question_text), 0,
                                         current_guesses, answer=correct, points=question_value, answerable=answerable))
            else:
                show_score(human + human_delta,
                           computer + computer_delta,
                           "HUMAN", "COMPUTER")
                print(format_display(display_num, question_text, ss, ii + 1,
                                     current_guesses, answer=correct,
                                     points=question_value, answerable=answerable))
    if computer_delta == 0:
        answer(final)
        if final == correct:
            return human + human_delta, computer + 10, final
        else:
            print("Incorrect answer: %s" % final)

    if human_delta == 0:
        response = None
        while response is None:
            os.system("afplay /System/Library/Sounds/Glass.aiff")
            response = input("Player, take a guess:\t")
            if '+' in response:
                return (human + 10,
                        computer + computer_delta,
                        response[1:])
            elif '-' in response:
                return (human, computer + computer_delta,
                        response[1:])
            else:
                response = None

    return human + human_delta, computer + computer_delta, ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--questions', type=str, default='questions.csv')
    parser.add_argument('--buzzes', type=str, default="ir_buzz.csv")
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--output', type=str, default="competition.csv")
    parser.add_argument('--finals', type=str, default="finals.csv")
    parser.add_argument('--power', type=str, default="power.csv")
    parser.add_argument('--max_questions', type=int, default=40)
    parser.add_argument('--readable', type=str, default="readable.txt")

    flags = parser.parse_args()

    questions = Questions(flags.questions)
    buzzes = Buzzes(flags.buzzes)
    finals = load_finals(flags.finals)
    power = PowerPositions(flags.power)
    qb_dataset = QuizBowlDataset(1)
    qb_answer_set = {format_guess(g) for g in qb_dataset.training_data()[1]}
    print("Done loading data")
    clear_screen()

    current_players = set()

    if True:
        print("Time for a buzzer check")
        players_needed = [1, 2, 3, 4]
        while len(current_players) < len(players_needed):
            print("Player %i, please buzz in" % min(x for x in players_needed \
                                                    if x not in current_players))
            press = interpret_keypress()
            if press in players_needed:
                os.system("afplay /System/Library/Sounds/Glass.aiff")
                print("Thanks for buzzing in, player %i!" % press)
                current_players.add(press)

        sleep(1.5)
        answer("I am ready too")

    human = 0
    computer = 0
    question_num = 0
    question_ids = sorted(questions._questions.keys(), key=lambda x: x % 10)

    question_ids = [x for x in question_ids if x in buzzes]

    if flags.readable != "":
        write_readable(flags.readable, questions, question_ids)

    skipped = 0
    for ii in question_ids:
        if skipped < flags.skip:
            skipped += 1
            continue
        
        question_num += 1
        power_mark = power(ii)
        if power_mark == "10":
            print("Looking for power for %i, got %s %s" %
                  (ii, power_mark, str(ii in power._power_marks.keys())))

        correct_answer = format_guess(questions.answer(ii))
        if correct_answer in qb_answer_set:
            answerable = 'answerable'
        else:
            answerable = 'not answerable'
        hum, comp, ans = present_question(question_num, ii, questions[ii],
                                          buzzes, finals[ii],
                                          questions.answer(ii),
                                          human=human,
                                          computer=computer,
                                          power=power(ii), answerable=answerable)
        human = hum
        computer = comp

        print("Correct answer of Question %i: %s" % (question_num,
                                                     questions.answer(ii)))
        sleep(kPAUSE)

        if question_num > flags.max_questions - 1:
            break

    show_score(human, computer,
               "HUMAN", "COMPUTER")

    if human == computer:
        print("Tiebreaker!")
        for ii in question_ids[question_num:]:
            question_num += 1
            hum, comp, ans = present_question(question_num, ii, questions[ii],
                                              buzzes, finals[ii],
                                              questions.answer(ii),
                                              human=human,
                                              computer=computer,
                                              power=power(ii))
            human = hum
            computer = comp

            print("Correct answer of Question %i: %s" % (question_num,
                                                         questions.answer(ii)))
            sleep(kPAUSE)

    show_score(human, computer, "HUMAN", "COMPUTER")
