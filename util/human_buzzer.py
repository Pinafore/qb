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

kSTATE = ["AB", "EB", "OB", "ES", "OS"]


def present_question(state, odd_score, even_score):
    assert state in kSTATE, "Invalid state %s" % state

    show_score(odd_score,
               even_score,
               "ODD TEAM", "EVEN TEAM",
               left_color="RED",
               right_color="YELLOW")

    if state == "AB":
        print("Free buzz!")
        press = interpret_keypress()
        if press == " ":
            present_question("AB", odd_score, even_score)
        if press % 2 == 0:
            os.system("say -v Tom Player %s" % press)
            present_question("ES", odd_score, even_score)
        if press % 2 == 1:
            os.system("say -v Tom Player %s" % press)
            present_question("OS", odd_score, even_score)

    elif state == "EB":
        print("Even buzz!")
        press = interpret_keypress()
        if press == " ":
            present_question("AB", odd_score, even_score)
        if press % 2 == 1:
            present_question("EB", odd_score, even_score)
        if press % 2 == 0:
            os.system("say -v Tom Player %s" % press)
            present_question("ES", odd_score, even_score)
    elif state == "OB":
        print("Odd buzz!")
        press = interpret_keypress()
        if press == " ":
            present_question("AB", odd_score, even_score)
        if press % 2 == 0:
            present_question("OB", odd_score, even_score)
        if press % 2 == 1:
            os.system("say -v Tom Player %s" % press)
            present_question("OS", odd_score, even_score)

    elif state == "ES" or state == "OS":
        score = -100
        while score < -5 or score > 45 or score % 5 != 0:
            score = raw_input("Score change:")
            try:
                score = int(score)
            except ValueError:
                score = -100

        if state == "ES":
            if score < 0:
                present_question("OB", odd_score, even_score + score)
            else:
                present_question("AB", odd_score, even_score + score)
        else:
            if score < 0:
                present_question("EB", odd_score + score, even_score)
            else:
                present_question("AB", odd_score + score, even_score)




if __name__ == "__main__":
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

    present_question("AB", 0, 0)
