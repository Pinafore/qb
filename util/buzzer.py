from __future__ import print_function
import textwrap
from collections import defaultdict
import argparse
from csv import DictReader
from time import sleep
from string import lower
from random import shuffle
import sys
import os

kSHOW_RIGHT = False
kPAUSE = .25

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


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


class PowerPositions:
    def __init__(self, filename):
        self._power_marks = {}
        try:
            infile = DictReader(open(filename, 'r'))
            for ii in infile:
                question = int(ii['question'])
                self._power_marks[question] = ii['word']
        except:
            print("Couldn't load from %s" % filename)
        print("Read power marks from %s: %s ..." %
              (filename, str(self._power_marks.keys())[1:69]))
        print(self._power_marks[700000020])

    def __call__(self, question):
        if question in self._power_marks:
            return self._power_marks[question]
        else:
            return "10"


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
               left_color="GREEN", right_color="BLUE"):
    clear_screen()

    # Print the header
    print("%-15s" % "")
    kCOLORS.print("%-15s" % left_header, left_color)
    print("%-30s" % "")
    kCOLORS.print("%-15s\n" % right_header, right_color)

    for line in xrange(1, 15):
        for num, color in [(left_score, left_color),
                           (right_score, right_color)]:
            for place in [100, 10, 1]:
                if place == 100 and num < 0:
                    val = -1
                else:
                    val = (abs(num) % (place * 10)) / place
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
        for ii in buzzfile:
            question, sent, word = int(ii["question"]), int(ii["sentence"]), int(ii["word"])
            if not (sent, word) in self._buzzes[question]:
                self._buzzes[question][(sent, word)] = {}
            self._buzzes[question][(sent, word)][ii["page"]] = \
              Guess(ii["page"], ii["evidence"], int(ii["final"]), float(ii["weight"]))

    def current_guesses(self, question, sent, word):
        try:
            ss, ww = max(x for x in self._buzzes[question] if
                            x[0] < sent or (x[0] == sent and x[1] <= word))
        except ValueError:
            return {}

        assert (ss, ww) in self._buzzes[question]
        return self._buzzes[question][(ss, ww)]

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
        for ii in qfile:
            self._questions[int(ii["id"])][int(ii["sent"])] = ii["text"]
            self._answers[int(ii["id"])] = ii["answer"].strip()

    def __iter__(self):
        for ii in self._questions:
            yield ii

    def __getitem__(self, val):
        return self._questions[val]

    def answer(self, val):
        return self._answers[val]


def format_display(display_num, question_text, sent, word, current_guesses,
                   answer=None, guess_limit=5, points=10):
    sep = "".join(["-"] * 80)

    current_text = ""
    for ss in xrange(sent):
        current_text += "%s " % question_text[ss]
    current_text += " ".join(question_text[sent].split()[:word])
    current_text = "\n".join(textwrap.wrap(current_text, 80))

    report = "Question %i: %i points\n%s\n%s\n%s\n\n" % \
        (display_num, points, sep, current_text, sep)

    for gg in sorted(current_guesses, key=lambda x: current_guesses[x].weight, reverse=True)[:guess_limit]:
        guess = current_guesses[gg]
        if guess.page == answer:
            report += "%s\t%f\t%s\n" % ("***CORRECT***", guess.weight, guess.evidence[:60])
        else:
            report += "%s\t%f\t%s\n" % (guess.page, guess.weight, guess.evidence[:60])
    return report

def load_finals(final_file):
    ff = DictReader(open(final_file))
    d = {}
    for ii in ff:
        d[int(ii['question'])] = ii['answer']
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
    try:
        press = int(press)
    except ValueError:
        press = None
    return press


def answer(ans):
    print("QANTA says:")
    os.system("afplay /System/Library/Sounds/Glass.aiff")
    os.system("say -v Tom %s" % ans.replace("'", ""))
    sleep(kPAUSE)
    print(ans)


def present_question(display_num, question_id, question_text, buzzes, final,
                     correct, human=0, computer=0, power="10"):

    human_delta = 0
    computer_delta = 0
    question_value = 15
    for ss in sorted(question_text):
        words = question_text[ss].split()
        for ii, ww in enumerate(words):
            if lower(ww).startswith(lower(power)):
                question_value = 10
            press = interpret_keypress()
            current_guesses = buzzes.current_guesses(question_id, ss, ii)
            buzz_now = [x for x in current_guesses.values() if x.final]
            assert len(buzz_now) < 2, "Cannot buzz on more than one thing"
            if press:
                os.system("afplay /System/Library/Sounds/Glass.aiff")
                response = None
                while response is None:
                    response = raw_input("Player %i, provide an answer:\t"
                                         %press)
                    if '+' in response:
                        return (human + question_value,
                                computer + computer_delta,
                                response[1:])
                    elif '-' in response:
                        if computer_delta == -5:
                            return (human, computer + computer_delta,
                                    response[1:])
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
                                     points=question_value))
                answer(buzz_now[0].page.split('(')[0])
                if buzz_now[0].page == correct:
                    return (human + human_delta, computer + question_value,
                            buzz_now[0].page)
                else:
                    computer_delta = -5
                    show_score(human + human_delta,
                               computer + computer_delta,
                               "HUMAN", "COMPUTER")
                    format_display(display_num, question_text,
                                   max(question_text), 0,
                                   current_guesses, answer=correct,
                                   points=question_value)
            else:
                show_score(human + human_delta,
                           computer + computer_delta,
                           "HUMAN", "COMPUTER")
                print(format_display(display_num, question_text, ss, ii + 1,
                                     current_guesses, answer=correct,
                                     points=question_value))
    if computer_delta == 0:
        answer(final.split('(')[0])
        if final == correct:
            return (human + human_delta, computer + 10, final)
        else:
            print("Incorrect answer: %s" % final)

    if human_delta == 0:
        response = None
        while response is None:
            response = raw_input("Player, take a guess:\t")
            if '+' in response:
                return (human + 10,
                        computer + computer_delta,
                        response[1:])
            elif '-' in response:
                return (human, computer + computer_delta,
                        response[1:])
            else:
                response = None

    return (human + human_delta, computer + computer_delta, "")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--questions', type=str, default='questions.csv')
    parser.add_argument('--buzzes', type=str, default="ir_buzz.csv")
    parser.add_argument('--output', type=str, default="competition.csv")
    parser.add_argument('--finals', type=str, default="finals.csv")
    parser.add_argument('--power', type=str, default="power.csv")
    parser.add_argument('--max_questions', type=int, default=40)

    flags = parser.parse_args()

    questions = Questions(flags.questions)
    buzzes = Buzzes(flags.buzzes)
    finals = load_finals(flags.finals)
    power = PowerPositions(flags.power)
    print("Done loading data")
    clear_screen()

    current_players = set()

    print("Time for a buzzer check")
    players_needed = [1]
    while len(current_players) < len(players_needed):
        print("Player %i, please buzz in" % min(x for x in players_needed \
                                                    if x not in current_players))
        press = interpret_keypress()
        if press in players_needed:
            os.system("afplay /System/Library/Sounds/Glass.aiff")
            print("Thanks for buzzing in, player %i!" % press)
            current_players.add(press)

    sleep(1.5)
    answer("I'm ready too")

    human = 0
    computer = 0
    question_num = 0
    question_ids = sorted(questions._questions.keys(), key=lambda x: x % 11)
    for ii in question_ids:
        question_num += 1
        power_mark = power(ii)
        if power_mark == "10":
            print("Looking for power for %i, got %s %s" %
                  (ii, power_mark, str(ii in power._power_marks.keys())))
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

    show_score(human, computer,
               "HUMAN", "COMPUTER")
