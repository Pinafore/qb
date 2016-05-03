from time import sleep
import pickle
import os

from util.buzzer import clear_screen, show_score
from util.buzzer import interpret_keypress

kSTATE = ["AB", "EB", "OB", "ES", "OS"]
kALLOWABLE_TOSSUP = "NPGC"
kALLOWABLE_BONUS = "ZFST"
kOUTPUT = "human_buzzer.csv"


def present_question(question, state, events, player=-1):
    assert state in kSTATE, "Invalid state %s" % state

    if question > 26:
        return

    odd_score = sum(events[x] for x in events if x[1] % 2 == 1)
    even_score = sum(events[x] for x in events if x[1] % 2 == 0)

    show_score(odd_score,
               even_score,
               "ODD TEAM", "EVEN TEAM",
               left_color="RED",
               right_color="YELLOW")
    print("Question %i" % question)

    if state == "AB":
        print("Free buzz!")
        press = interpret_keypress()
        if press is None:
            present_question(question, state, events)
        if press == " ":
            present_question(question + 1, "AB", events)
        if press % 2 == 0:
            os.system("say -v Tom Player %s" % press)
            present_question(question, "ES", events, press)
        if press % 2 == 1:
            os.system("say -v Tom Player %s" % press)
            present_question(question, "OS", events, press)

    elif state == "EB":
        print("Even buzz!")
        press = interpret_keypress()
        if press is None:
            present_question(question, state, events)
        if press == " ":
            present_question(question + 1, "AB", events)
        if press % 2 == 1:
            present_question(question, "EB", events)
        if press % 2 == 0:
            os.system("say -v Tom Player %s" % press)
            present_question(question, "ES", events, press)
    elif state == "OB":
        print("Odd buzz!")
        press = interpret_keypress()
        if press is None:
            present_question(question, state, events)
        if press == " ":
            present_question(question + 1, "AB", events)
        if press % 2 == 0:
            present_question(question, "OB", events)
        if press % 2 == 1:
            os.system("say -v Tom Player %s" % press)
            present_question(question, "OS", events, press)

    elif state == "ES" or state == "OS":
        assert player != -1, "invalid player"
        tossup = '0'
        while not tossup in kALLOWABLE_TOSSUP:
            print("Tossup %i result [%s]:" % (question, kALLOWABLE_TOSSUP),
                  end='')
            tossup = interpret_keypress(kALLOWABLE_TOSSUP)

        if tossup == "P":
            tossup = 15
        elif tossup == "N":
            tossup = -5
        elif tossup == "G":
            tossup = 0
        elif tossup == "C":
            tossup = 10

        if tossup <= 0:
            if any(x[0] == question for x in events):
                # Both have buzzed
                events[(question, player, "TU")] = 0
                present_question(question + 1, "AB", events)
            else:
                # Nobody buzzed
                events[(question, player, "TU")] = tossup
                present_question(question, "EB" if state == "OS" else "OB",
                                 events)
        else:
            events[(question, player, "TU")] = tossup
            odd_score = sum(events[x] for x in events if x[1] % 2 == 1)
            even_score = sum(events[x] for x in events if x[1] % 2 == 0)

            show_score(odd_score,
                       even_score,
                       "ODD TEAM", "EVEN TEAM",
                       left_color="RED",
                       right_color="YELLOW")

            bonus = "0"
            while not bonus in kALLOWABLE_BONUS:
                print("Bonus %i result [%s]:" % (question, kALLOWABLE_BONUS),
                      end='')
                bonus = interpret_keypress(kALLOWABLE_BONUS)

            if bonus == "Z":
                events[(question, player, "BONUS")] = 0
            elif bonus == "F":
                events[(question, player, "BONUS")] = 10
            elif bonus == "S":
                events[(question, player, "BONUS")] = 20
            elif bonus == "T":
                events[(question, player, "BONUS")] = 30

            pickle.dump(events, open(kOUTPUT, 'w'))
            present_question(question + 1, "AB", events)


def main():
    clear_screen()

    current_players = set()

    if True:
        print("Time for a buzzer check")
        players_needed = list(range(2, 9, 2)) + list(range(1, 9, 2))
        while len(current_players) < len(players_needed):
            print("Player %i, please buzz in" %
                  [x for x in players_needed if x not in current_players][0])
            press = interpret_keypress()
            if press in players_needed:
                os.system("say -v Tom Player %i!" % press)
                current_players.add(press)

        sleep(1.5)

    if os.path.isfile(kOUTPUT):
        events = pickle.load(open(kOUTPUT))
        question = max(x[0] for x in events)
    else:
        events = {}
        question = 1

    present_question(question, "AB", events)

if __name__ == "__main__":
    main()
