from __future__ import print_function
import textwrap
from collections import defaultdict
import argparse
from csv import DictReader
from time import sleep
from random import shuffle
import sys
import os
import random

from buzzer import kSHOW_RIGHT, kPAUSE, kBIGNUMBERS
from buzzer import clear_screen, PowerPositions, show_score
from buzzer import Guess, Buzzes, Questions, format_display
from buzzer import interpret_keypress, answer, create_parser
from buzzer import load_data, buzzer_check, question_loop, Score


def present_question_hh(display_num, question_id, question_text, buzzes, final,
                        correct, score, power="10"):

    even_delta = 0
    odd_delta = 0
    question_value = 15

    final_system = random.choice(list(final.keys()))
    final_answer = final[final_system]

    # Find out where the computer would buzz
    computer_position = (max(question_text) + 1, 0)
    for ss in sorted(question_text):
        words = question_text[ss].split()
        for ii, ww in enumerate(words):
            current_guesses = buzzes.current_guesses(question_id, ss, ii - 1)
            buzz_now = [x for x in current_guesses.values() if x.final]
            if len(buzz_now) > 0:
                computer_position = ss, ii
                computer_guess = buzz_now[0].page
                computer_system = buzz_now[0].system
                break

    question_text[ss] += "~ ~ ~ ~ ~"

    question_done = False
    human_delta = 0
    computer_delta = 0
    for ss in sorted(question_text):
        words = question_text[ss].split()
        for ii, ww in enumerate(words):
            if question_done:
                break

            if computer_position[0] == ss and computer_position[1] == ii:
                # This is where the computer buzzes
                if human_delta <= 0:
                    if computer_guess == correct:
                        # os.system("afplay sounds/applause.wav")
                        if human_delta <= 0:
                            computer_delta = question_value
                    else:
                        # answer(computer_guess, computer_system)
                        # os.system("afplay sounds/sad_trombone.wav")
                        if human_delta == 0:
                            computer_delta = -5
                else:
                    # answer(computer_guess, computer_system)
                    question_done = True
                    computer_delta = 0

            current_guesses = buzzes.current_guesses(question_id, ss, ii)
            if ww.lower().startswith(power.lower()):
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

                os.system("afplay /System/Library/Sounds/Glass.aiff")
                response = None
                while response is None:
                    response = input("Player %i, provide an answer:\t"
                                     % press)
                    if '+' in response:
                        if press % 2 == 0:
                            even_delta = question_value
                        else:
                            odd_delta = question_value
                        if computer_delta < 0 and human_delta == 0:
                            human_delta = question_value
                            question_done = True
                        elif computer_delta == 0:
                            human_delta = question_value
                    elif '-' in response:
                        if even_delta == 0 and press % 2 != 0:
                            odd_delta = -5
                        if odd_delta == 0 and press % 2 == 0:
                            even_delta = -5
                        if computer_delta < 0:
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
                print(human_delta, computer_delta, even_delta, odd_delta)

                print(format_display(display_num, question_text, ss, ii + 1,
                                     current_guesses, answer=correct,
                                     points=question_value))

    # Now see what the computer would do
    if computer_delta == 0 and human_delta <= 0:
        # answer(final_answer.split('(')[0], final_system)
        if final == correct:
            computer_delta = 10
        else:
            print("Computer guesses incorrectly: %s" % final)
    elif computer_delta > 0:
        # answer(computer_guess, computer_system)
        format_display(display_num, question_text, computer_position[0],
                       computer_position[1], current_guesses, answer=correct,
                       points=computer_delta)

    return(Score(even_delta, odd_delta, human_delta, computer_delta))

def check_hh_tie(score):
    """
    For the computer-human and human-human programs, this needs to be
    different.  This is why it's a silly function.
    """
    return score.even == score.odd


if __name__ == "__main__":
    flags = create_parser()
    questions, buzzes = load_data(flags)
    print("Done loading data")
    
    clear_screen()
    buzzer_check(flags.players)

    score = question_loop(flags, questions, buzzes, present_question_hh, 
                          check_hh_tie)

    show_score(score.human, score.computer,
               "HUMAN", "COMPUTER")
            
