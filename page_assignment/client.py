#!/usr/bin/env python
import curses
from time import sleep
import pickle
import datetime
from collections import defaultdict
from itertools import chain
from csv import DictWriter, DictReader
from glob import glob
import os
import re
from os import system
import string
from string import ascii_lowercase

from fuzzywuzzy import process
from page_assignment.active_learning_for_matching import ActiveLearner, simple_menu
from util.qdb import QuestionDatabase
from unidecode import unidecode

valid_strings = set(ascii_lowercase) | set(str(x) for x in xrange(10)) | set([' '])
paren_expression = re.compile('\s*\([^)]*\)\s*')
brackets = re.compile(r'\[[^)]*\]')


def normalize(text):
        text = brackets.sub("", text)
        text = paren_expression.sub(" ", text)
        text = unidecode(text).lower()
        text = text.replace("{", "").replace("~", "").replace("}", "")
        " ".join(x for x in text.split())
        return ''.join(x for x in text if x in valid_strings)

class TitleFinder:
    def __init__(self, index):
        print("Loading big index ...")
        temp_index = pickle.load(index)
        print("done")
        self._index = defaultdict(set)

        print("Creating smaller index ...")
        for title in temp_index:
            for word in normalize(title).split():
                self._index[word].add(title)
        print("done")

    def query(self, text, num_candidates=25, max_word_hits=1000):
        tokens = normalize(text).split()
        candidates = chain.from_iterable(self._index[x] for x in tokens \
          if len(self._index[x]) < max_word_hits)

        return process.extract(text, set(candidates), limit=num_candidates)

class Answer:
    def __init__(self, user, question, sentence, word, last, overall, text, correct):
        self.user = user
        self.question = question
        self.sentence = sentence
        self.word = word
        self.last = last
        self.overall = overall
        self.text = text
        self.correct = correct

def incremental_query(win, words):
  win.nodelay(True) # make getkey() not wait
  x = 0
  while x <= len(words):
    #just to show that the loop runs, print a counter
    win.clear()
    win.addstr(0,0," ".join(words[:x]))
    x += 1

    try:
      key = win.getkey()
    except: # in no delay mode getkey raise and exeption if no key is press
      key = None
    if key == " ": # of we got a space then break
      break
    sleep(.2)

  return max(0, x - 2)

def get_answer(sentences, answer, page):
    words = []
    positions = {}
    index = 0

    for ii, ss in enumerate(sentences):
        for jj, ww in enumerate(ss.split()):
            words.append(ww)
            positions[index] = (ii, jj)
            index += 1

    word_position = curses.wrapper(incremental_query, words)
    print " ".join(words[:word_position])
    print()
    guess = raw_input("ANSWER>")

    system("say %s" % "".join(x for x in answer if not x in string.punctuation))

    return word_position, positions[word_position], guess

class PerformanceWriter:
    def __init__(self, location, user):
        stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self._handle = open("%s/%s_%s.csv" % (location, user, stamp), 'w')
        fields = ["user", "time", "question", "index", "sentence", "word", "guess", "answer", "page"]
        self._csv = DictWriter(self._handle, fieldnames=dict((x, x) for x in fields))
        self._csv.writeheader()
        self._user = user

    def add_row(self, question, index, sentence, word, guess, answer, page):
        stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        row = {}
        row['user'] = self._user
        row['time'] = stamp
        row['question'] = question
        row['index'] = index
        row['sentence'] = sentence
        row['word'] = word
        row['guess'] = guess
        row['answer'] = answer
        row['page'] = unidecode(page)

        self._csv.writerow(row)
        self._handle.flush()


def already_answered(path, user):
    seen = set()
    search = "%s/%s*.csv" % (path, user)
    print(search)
    for ii in glob(search):
        print(ii)
        with open(ii) as infile:
            for jj in DictReader(infile):
                seen.add(int(jj['question']))
                print(jj)
    print(seen)
    sleep(1)
    return seen

if __name__ == "__main__":
    from util import flags

    flags.define_string("title_index", None, "Pickle of all titles")
    flags.define_string("label_path", None, "Where we write page associations")
    flags.define_string("database", None, "Question database")
    flags.define_string("performance_output", None, "Where we write user performance")
    flags.define_string("user", None, "User identifier")
    flags.InitFlags()

    seen = already_answered(flags.performance_output, flags.user)
    al = ActiveLearner(None, flags.label_path)
    print("Loading question db %s" % flags.database)
    db = QuestionDatabase(flags.database)
    pw = PerformanceWriter(flags.performance_output, flags.user)
    tf = TitleFinder(open(flags.title_index))


    questions = db.questions_by_tournament("High School Championship")
    for qid in questions:
        question = questions[qid]
        if question.fold == "train" or qid in seen:
            continue
        choices = list(tf.query(question.answer))

        # Get what and when the human answered
        wp, idx, ans = get_answer([question.text[x] for x in sorted(question.text)], question.answer, question.page)

        print("\n".join(question.text.values()))
        print("\n")
        print("--------------------\n")
        print(question.answer, question.page)
        print("--------------------\n")
        sleep(1)
        if question.page == "":
            page = simple_menu([x[0] for x in choices], tf._index, [x[1] for x in choices])
            al.remember(question.qnum, page)

            al.dump(flags.label_path)
        else:
            page = question.page
        pw.add_row(question.qnum, wp, idx[0], idx[1], ans, question.answer, page)
