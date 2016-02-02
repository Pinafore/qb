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

def present_question(display_num):
    even_delta = 0
    odd_delta = 0
    press = interpret_keypress()
    while press != "direction":
        if press:
            # Check to see if buzz is valid
            if even_delta != 0 and press % 2 == 0:
                continue
            if odd_delta != 0 and press % 2 != 0:
                continue
     
            os.system("afplay /System/Library/Sounds/Glass.aiff")
            response = None
            while response is None:
                    os.system("say -v Tom Player %s" % press)
                    response = raw_input("Player %i, provide an answer:\t"
                                         % press)
                    if '+' in response:
                        if press % 2 == 0:
                            even_delta = int(response.split("+")[1])
                        else:
                            odd_delta = int(response.split("+")[1])
                    elif '-' in response:
                        if even_delta == 0 and press % 2 != 0:
                            odd_delta = -5
                        if odd_delta == 0 and press % 2 == 0:
                            even_delta = -5

                        # Break if both teams have answered
                        if even_delta != 0 and press % 2 != 0:
                            question_done = True
                        if odd_delta != 0 and press % 2 == 0:
                            question_done = True
                    else:
                        response = None
        print("Waiting for buzz")
        press = interpret_keypress()
        print(press)

    return even_delta, odd_delta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--max_questions', type=int, default=26)

    flags = parser.parse_args()    
    clear_screen()

    current_players = set(range(8))

    if False:
        print("Time for a buzzer check")
        players_needed = [1]
        while len(current_players) < len(players_needed):
            print("Player %i, please buzz in" % min(x for x in players_needed \
                                                    if x not in current_players))
            press = interpret_keypress()
            if press in players_needed:
                os.system("afplay /System/Library/Sounds/Glass.aiff")
                answer("Thanks for buzzing in, player %i!" % press)
                current_players.add(press)

        sleep(1.5)

    question_num = 0
    for ii in range(flags.max_questions):
        # print(ii, questions[ii])
        question_num += 1
        score = present_question(question_num)

        show_score(score[0],
                   score[1],
                   "TEAM A", "TEAM B",
                   left_color="RED",
                   right_color="YELLOW")

        if question_num > flags.max_questions - 1:
            break


