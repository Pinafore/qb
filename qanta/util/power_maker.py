import textwrap
from collections import defaultdict
import argparse
from csv import DictReader, DictWriter
from time import sleep
import sys
import os


if __name__ == '__main__':

    qfile = DictReader(open('results/expo/questions.csv', 'r'))
    power_mark = '(*)'
    header = ['question', 'sent', 'word']
    pfile = DictWriter(open('data/expo_power.csv', 'w'), header)
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
    		for index, token in enumerate(text.split()):
    			if token == power_mark:
    				power_index = index
    				break
    		if power_index != 0:
    			power_index -= 1
    		out['word'] = power_index
    		print text, power_index, ii['sent']
    		pfile.writerow(out)



