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

from buzzer import kSHOW_RIGHT, kPAUSE, kBIGNUMBERS
from buzzer import clear_screen, PowerPositions, show_score
from buzzer import Guess, Buzzes, Questions, format_display
from buzzer import load_finals, interpret_keypress, answer

kPADDING = "WAIT"
kPAD_LEN = 5

class Score:
    def __init__(self, even=0, odd=0, human=0, computer=0):
        self.even = even
        self.odd = odd
        self.human = human
        self.computer = computer

    def add(self, score):
        return Score(self.even + score.even,
                     self.odd + score.odd,
                     self.human + score.human,
                     self.computer + score.computer)


def present_question(display_num, question_id, question_text, buzzes, final,
                     correct, score, power="10"):

    even_delta = 0
    odd_delta = 0
    question_value = 15

    # Find out where the computer would buzz
    computer_position = (max(question_text) + 1, 0)
    for ss in sorted(question_text):
        words = question_text[ss].split()
        for ii, ww in enumerate(words):
            current_guesses = buzzes.current_guesses(question_id, ss, ii)
            buzz_now = [x for x in current_guesses.values() if x.final]
            if len(buzz_now) == 1:
                computer_position = ss, ii
                computer_guess = buzz_now[0].page
                break

    question_text[ss] += " %s " % " ".join([kPADDING] * 5)

    question_done = False
    human_delta = 0
    computer_delta = 0
    human_answers = 0
    computer_answers = 0
    for ss in sorted(question_text):
        words = question_text[ss].split()
        for ii, ww in enumerate(words):
            if question_done or human_answers==2:
                break

            if computer_position[0] == ss and computer_position[1] == ii:
                # This is where the computer buzzes
                if human_delta <= 0:
                    if computer_guess == correct:
                        if human_answers == 2:
                            answer(computer_guess.split("(")[0])
                            computer_answers += 1
                            question_done = True
                        os.system("afplay sounds/applause.wav")
                        if human_delta <= 0:
                            computer_delta = question_value
                    else:
                        answer(computer_guess.split("(")[0])
                        computer_answers += 1
                        os.system("afplay sounds/sad_trombone.wav")
                        if human_delta == 0:
                            computer_delta = -5
                else:
                    answer(computer_guess.split("(")[0])
                    computer_answers += 1
                    question_done = True
                    computer_delta = 0

            current_guesses = buzzes.current_guesses(question_id, ss, ii)
            if lower(ww).startswith(lower(power)):
                question_value = 10
            press = interpret_keypress()

            if isinstance(press, int):
                # Check to see if buzz is valid
                if human_delta > 0:
                    continue
                if even_delta != 0 and press % 2 == 0:
                    continue
                if odd_delta != 0 and press % 2 != 0:
                    continue
                human_answers += 1

                os.system("afplay /System/Library/Sounds/Glass.aiff")
                response = None
                while response is None:
                    response = raw_input("Player %i, provide an answer:\t"
                                         % press)
                    if '+' in response:
                        if press % 2 == 0:
                            even_delta = question_value
                        else:
                            odd_delta = question_value
                        if computer_delta < 0 and human_delta == 0:
                            human_delta = question_value
                            question_done = True
                        elif computer_delta == 0 and human_delta == 0:
                            human_delta = question_value
                    elif '-' in response:
                        if even_delta == 0 and press % 2 != 0 and ww != kPADDING:
                            odd_delta = -5
                        if odd_delta == 0 and press % 2 == 0 and ww != kPADDING:
                            even_delta = -5
                        if computer_delta == 0:
                            human_delta = -5

                        # Break if both teams have answered
                        if even_delta != 0 and press % 2 != 0:
                            question_done = True
                        if odd_delta != 0 and press % 2 == 0:
                            question_done = True
                    else:
                        response = None
            # Don't buzz if anyone else has gotten it wrong
            else:
                show_score(score.even + even_delta,
                           score.odd + odd_delta,
                           "TEAM A", "TEAM B",
                           left_color="RED",
                           right_color="YELLOW")
                show_score(score.human + human_delta,
                           score.computer + computer_delta,
                           "HUMAN", "COMPUTER",
                           flush=False)


                print(format_display(display_num, question_text, ss, ii + 1,
                                     current_guesses, answer=correct,
                                     points=question_value))

    # Now see what the computer would do
    if computer_delta == 0 and human_delta <= 0 and computer_answers == 0:
        show_score(score.even + even_delta,
                   score.odd + odd_delta,
                   "TEAM A", "TEAM B",
                   left_color="RED",
                   right_color="YELLOW")
        show_score(score.human + human_delta,
                   score.computer + computer_delta,
                   "HUMAN", "COMPUTER",
                   flush=False)
        print(format_display(display_num, question_text, max(question_text) - 1,
                             1000, current_guesses, answer=correct,
                             points=computer_delta))
        answer(final.split('(')[0])
        if final == correct:
            computer_delta = 10
        else:
            print("Computer guesses incorrectly: %s" % final)
    elif computer_delta > 0 and computer_answers == 0:
        format_display(display_num, question_text, computer_position[0],
                       computer_position[1], current_guesses, answer=correct,
                       points=computer_delta)
        answer(computer_guess)

    return score.add(Score(even_delta, odd_delta, human_delta, computer_delta))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--questions', type=str, default='questions.csv')
    parser.add_argument('--buzzes', type=str, default="ir_buzz.csv")
    parser.add_argument('--output', type=str, default="competition.csv")
    parser.add_argument('--finals', type=str, default="finals.csv")
    parser.add_argument('--power', type=str, default="power.csv")
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--max_questions', type=int, default=20)
    parser.add_argument('--readable', type=str, default="readable.txt")

    flags = parser.parse_args()

    questions = Questions(flags.questions)
    buzzes = Buzzes(flags.buzzes)
    finals = load_finals(flags.finals)
    power = PowerPositions(flags.power)
    print("Done loading data")
    clear_screen()

    current_players = set()

    if False:
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

    score = Score()
    question_ids = sorted(questions._questions.keys(), key=lambda x: x % 11)

    if flags.readable != "":
        question_num = 0
        o = open(flags.readable, 'w')
        for ii in question_ids:
            question_num += 1
            o.write("%i) " % question_num)
            for jj in questions[ii]:
                o.write("%s  " % questions[ii][jj])
            o.write("\nANSWER: %s\n\n" % questions.answer(ii))

    question_num = 0
    question_ids = list(question_ids)[flags.skip:]
    for ii in question_ids:
        # print(ii, questions[ii])
        question_num += 1
        power_mark = power(ii)
        score = present_question(question_num, ii, questions[ii],
                                 buzzes, finals[ii],
                                 questions.answer(ii),
                                 score=score,
                                 power=power(ii))

        print("Correct answer of Question %i: %s" % (question_num,
                                                     questions.answer(ii)))
        sleep(kPAUSE)

        if question_num > flags.max_questions - 1:
            break

    show_score(score.even,
               score.odd,
               "TEAM A", "TEAM B",
               left_color="RED",
               right_color="YELLOW")

    show_score(score.human, score.computer)
