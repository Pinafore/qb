from __future__ import print_function
import textwrap
from collections import defaultdict
import argparse
import random
from csv import DictReader
from time import sleep
import datetime
#from str import lower
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

def write_readable(filename, ids, questions):
    question_num = 0
    o = open(flags.readable, 'w')
    for ii in question_ids:
        question_num += 1
        o.write("%i) " % question_num)
        power_found = False
        for jj in questions[ii]:
            if questions._power(ii) and not power_found and questions._power(ii).lower() in questions[ii][jj].lower():
                power_found = True
                o.write("%s  " %
                        questions[ii][jj].replace(power(ii), "(*) %s" %
                                                  power(ii)))
            else:
                o.write("%s  " % questions[ii][jj])
        o.write("\nANSWER: %s\n\n" % questions.answer(ii))

def clear_screen():
    print("Clearing")
    os.system('cls' if os.name == 'nt' else 'clear')


class PowerPositions:
    def __init__(self, filename):
        self._power_marks = {}
        if filename:
            try:
                infile = DictReader(open(filename, 'r'))
                for ii in infile:
                    question = int(ii['question'])
                    self._power_marks[question] = ii['word']
            except:
                print("Couldn't load from %s" % filename)
            print("Read power marks from %s: %s ..." %
                (filename, str(self._power_marks.keys())[1:69]))
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
    def __init__(self, system, page, evidence, final, weight):
        self.system = system
        self.page = page.replace("_", " ")
        self.evidence = evidence
        self.final = final
        self.weight = weight

class Buzzes:
    def __init__(self, file_path):
        self._buzzes = defaultdict(dict)
        self._finals = defaultdict(dict)
        print("Initializing buzz files")

    def debug(self):
        self.add_guess(0, 0, 5, "A", "Heisenberg", "", 0, .2)
        self.add_guess(0, 0, 5, "C", "Narcos", "", 0, .2)
        self.add_guess(0, 2, 3, "A", "Better_Call_Saul", "", 1, .7)
        self.add_guess(0, 2, 3, "B", "Breaking Bad", "", 1, .6)
        self.add_guess(0, 2, 3, "C", "Breaking Bad", "", 1, .5)
        self._finals[0]["A"] = "Breaking Bad"
        self._finals[0]["B"] = "Breaking Bad"
        self._finals[0]["C"] = "Breaking Bad"        

        self.add_guess(1, 0, 5, "A", "SimCity", "", 0, .2)
        self.add_guess(1, 0, 5, "C", "Skyrim", "", 0, .2)
        self.add_guess(1, 2, 3, "A", "Skyrim", "", 0, .7)
        self.add_guess(1, 3, 3, "B", "Jedi Knight", "", 0, .6)
        self.add_guess(1, 3, 3, "C", "Jedi Knight", "", 0, .5)
        self._finals[1]["A"] = "Fallout 76"
        self._finals[1]["B"] = "Fallout 76"
        self._finals[1]["C"] = "Fallout (series)"        

        self.add_guess(2, 0, 5, "A", "Apple", "", 0, .2)
        self.add_guess(2, 0, 5, "C", "Onion", "", 0, .2)
        self.add_guess(2, 0, 5, "C", "Apple", "", 0, .2)
        self.add_guess(2, 2, 3, "A", "Apple", "", 1, .7)
        self.add_guess(2, 3, 3, "B", "Onion", "", 0, .6)
        self.add_guess(2, 3, 3, "C", "Jedi Knight", "", 0, .5)
        self._finals[2]["A"] = "Potato"
        self._finals[2]["B"] = "Potato"
        self._finals[2]["C"] = "Potato"  

    def add_guess(self, question, sent, word, system, guess, evidence, final, weight):
        if not (sent, word) in self._buzzes[question]:
            self._buzzes[question][(sent, word)] = {}
        self._buzzes[question][(sent, word)][guess] = \
            Guess(system,guess, evidence, final, weight)
        
    def add_system(self, file_path):
        self.load_finals("%s.finals" % file_path)
        
        buzzfile = DictReader(open("%s.buzz" % buzz_file, 'r'))
        system = buzzfile.split('/')[-1]
        system = buzzfile.split('.')[0]
        system = buzzfile.split("_")[0]

        for ii in buzzfile:
            question, sent, word = int(ii["question"]), int(ii["sentence"]), int(ii["word"])
            self.add_guess(question, sent, word, system, ii["page"], ii["evidence"],
                           int(ii["final"]), float(ii["weight"]))

    def load_finals(self, system, final_file):
        ff = DictReader(open(final_file))
        for ii in ff:
            self._finals[int(ii['question'])][system] = ii['answer'].replace('_', ' ')

            
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

class Questions:
    def __init__(self):
        self._questions = defaultdict(dict)
        self._answers = defaultdict(str)
        self._power = PowerPositions("")
        print("Initializing questions")

    def debug(self):
        self._questions[0] = {0: "His aliases include: Pastor Hansford of Free Will Baptist in Coushatta, Louisianna; Viktor with a K St. Claire, a South African who just inherited money from his uncle; Charlie Hustle, a mailboy in the HHM mailroom; and Gene Takavic, a Cinnabon manager in Omaha.",
                              1: "His best known alias came from selling burner phones in Albuquerque, New Mexico and was more fitting for a lawyer than his birthname, James McGill.",
                              2: "For ten points, name this titular CRIMINAL lawyer from an AMC series who originated on Breaking Bad."}
        self._answers[0] = "Better Call Saul"

        self._questions[1] = {0: 'This game contains an easter egg based on fan art of blocks spelling out "P A M" and an insect in a top hat, and its launch was declared a holiday by Jim Justice on November 14, 2018.',  
                              1: 'The "Personal Matters" quest requires the player to kill Evan, who was the fiancé of the Overseer.  Locations in this game include the Whitespring Resort, which is based on the real-life Greenbrier resort in Solphur Springs, West Virginia.',
                              2: "A multiplayer game centered on a control vault, for ten points name Bathesda's most recent entry in the Fallout franchise."}
        self._answers[1] = "Fallout 76"

        self._questions[2] = {0: "Amsterdam experienced a riot named for this crop during World War One.",
                              1: "In the 1830s and 40s, Russian peasants on the lower Volga objected to the forced introduction of this crop in a namesake series of riots.",
                              2: "Antoine-Augustin Parmentier pushed for the cultivation of this crop in France.",
                              3: "This crop provides an alternate name for the War of the Bavarian Succession.",
                              4: "An epidemic of phytophthora infestans devastated the cultivation of this crop in the mid-19th Century.",
                              5: "Robert Peel repealed the Corn Laws to counteract a blight of, for 10 points, what crop that caused an Irish famine?"}
        self._answers[2] = "Potato"

    def load_power(self, power_file):
        self._power = PowerPositions(power_file)
    
    def load_questions(self, question_file):
        qfile = DictReader(open(question_file, 'r'))

        for ii in qfile:
            self._questions[int(ii["id"])][int(ii["sent"])] = ii["text"]
            self._answers[int(ii["id"])] = ii["answer"].strip().replace("_", " ")

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
    for ss in range(sent):
        current_text += "%s " % question_text[ss]
    current_text += " ".join(question_text[sent].split()[:word])
    current_text = "\n".join(textwrap.wrap(current_text, 80))

    report = "Question %i: %i points\n%s\n%s\n%s\n\n" % \
        (display_num, points, sep, current_text, sep)

    for gg in sorted(current_guesses, key=lambda x: current_guesses[x].weight, reverse=True)[:guess_limit]:
        guess = current_guesses[gg]
        if guess.page == answer:
            report += "%-10s\t%-50s\t%0.2f\t%s\n" % (guess.system, "***CORRECT***", guess.weight, guess.evidence[:60])
        else:
            report += "%-10s\t%-50s\t%0.2f\t%s\n" % (guess.system, guess.page, guess.weight, guess.evidence[:60])
    return report



def interpret_keypress(other_allowable=""):
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
    print(ans)


def present_question(display_num, question_id, question_text, buzzes, final,
                     correct, human=0, computer=0, power="10"):

    human_delta = 0
    computer_delta = 0
    question_value = 15
    for ss in sorted(question_text):
        words = question_text[ss].split()
        for ii, ww in enumerate(words):
            # if we've reached the end of the question
            if ss == max(question_text) and ii == len(question_text[ss].split()) - 1:
                 # If humans haven't buzzed, let the computer buzz
                 if computer_delta == 0:
                    print(final)
                    system = random.choice(list(final.keys()))
                    answer(final[system].split('(')[0], system)
                    final = final[system]
                    if final == correct:
                        return (human + human_delta, computer + 10, final)
                    else:
                        print("Incorrect answer: %s" % final)
                 else:
                     words += [" ", " ", " ", " ", " "] 
            
            if ww.lower().startswith(power.lower()):
                question_value = 10
            press = interpret_keypress()
            current_guesses = buzzes.current_guesses(question_id, ss, ii - 2)
            buzz_now = [x for x in current_guesses.values() if x.final]
            # Removing this assertion now that we can have multiple systems playing
            # assert len(buzz_now) < 2, "Cannot buzz on more than one thing"
            if isinstance(press, int):
                os.system("afplay /System/Library/Sounds/Glass.aiff")
                response = None
                while response is None:
                    response = input("Player %i, provide an answer:\t"
                                         % press)
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
                answer(buzz_now[0].page.split('(')[0], buzz_now[0].system)
                if buzz_now[0].page == correct:
                    print("Computer guesses: %s (correct)" % buzz_now[0].page)
                    sleep(1)
                    return (human + human_delta, computer + question_value,
                            buzz_now[0].page)
                else:
                    print("Computer guesses: %s (wrong)" % buzz_now[0].page)
                    sleep(1)
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

    return (human + human_delta, computer + computer_delta, "")

def create_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--questions', type=str, default='')
    parser.add_argument('--model_directory', type=str, default="")
    parser.add_argument('--model_name', type=str, default="")    
    parser.add_argument('--output', type=str, default="GAMEPLAY %s.csv" % datetime.datetime.now().strftime("%I:%M%p on %B %d %Y"))
    parser.add_argument('--players', type=int, default=1)
    parser.add_argument('--human_start', type=int, default=0)
    parser.add_argument('--computer_start', type=int, default=0)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--power', type=str, default="")
    parser.add_argument('--max_questions', type=int, default=40)
    parser.add_argument('--readable', type=str, default="readable.txt")
    return parser.parse_args()

def load_data(flags):
    questions = Questions()
    buzzes = Buzzes(flags.model_directory)
    
    if flags.questions != "":
        questions.load_questions(flags.questions)
        questions.load_power(flags.power)
    else:
        questions.debug()

    if flags.model_directory != "":
        for ii in glob("%s/*" % flags.model_drectory):
            if ii.endswith(".buzz.csv"):
                # if we have a specific model, only load that
                if flags.model != "" and "%s.buzz.csv" % flags.model not in ii:
                    print("Not loading model %s" % ii)
                else:
                    buzzes.add_system(ii.replace(".buzz.csv", ""))
    else:
        buzzes.debug()

    return questions, buzzes

def buzzer_check(players):
    if players > 0:
        print("Time for a buzzer check")
    players_needed = range(1, players + 1)
    current_players = set()
    while len(current_players) < len(players_needed):
        print("Player %i, please buzz in" % min(x for x in players_needed \
                                                if x not in current_players))
        press = interpret_keypress()
        if press in players_needed:
            os.system("afplay /System/Library/Sounds/Glass.aiff")
            print("Thanks for buzzing in, player %i!" % press)
            current_players.add(press)
 
    if players > 0:
        sleep(1.5)
        answer("I'm ready too", "QANTA")

if __name__ == "__main__":
    flags = create_parser()
    questions, buzzes = load_data(flags)
    print("Done loading data")
    
    clear_screen()
    buzzer_check(flags.players)

    human = flags.human_start
    computer = flags.computer_start
    question_num = 0
    question_ids = sorted(questions._questions.keys(), key=lambda x: x % 10)

    question_ids = [x for x in question_ids if x in buzzes]

    if flags.readable != "":
        write_readable(flags.readable, question_ids, questions)

    for ii in question_ids:
        question_num += 1
        if flags.skip > 0 and question_num < flags.skip:
            continue

        power_mark = questions._power(ii)
        if power_mark == "10":
            print("Looking for power for %i, got %s %s" %
                  (ii, power_mark, str(ii in power._power_marks.keys())))
        hum, comp, ans = present_question(question_num, ii, questions[ii],
                                          buzzes, buzzes._finals[ii],
                                          questions.answer(ii),
                                          human=human,
                                          computer=computer,
                                          power=questions._power(ii))
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
