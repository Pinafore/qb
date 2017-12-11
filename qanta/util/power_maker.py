import textwrap
from collections import defaultdict
import argparse
from csv import DictReader, DictWriter
from time import sleep
import sys
import os


if __name__ == '__main__':

    qfile = DictReader(open('data/questions/expo/questions.csv', 'r'))
    power_mark = '(*)'
    header = ['question', 'sent', 'word']
    pfile = DictWriter(open('data/questions/expo/questions.power.csv', 'w'), header)
    pfile.writeheader()

    questions = defaultdict(dict)
    answers = defaultdict(str)
    for ii in qfile:
      text = ii['text']
      # power detected!
      if power_mark in text:
        out = {}
        out['question'] = ii['id']
        out['sent'] = ii['sent']
        power_index = -1
        tokens = text.split()
        for index, token in enumerate(tokens):
          if token == power_mark:
            power_index = index
            break
        if power_index == len(tokens) - 1:
          out['word'] = '???'
          print(text, power_index, ii['sent'])
        else:
          out['word'] = tokens[power_index + 1]
          
        
        pfile.writerow(out)



