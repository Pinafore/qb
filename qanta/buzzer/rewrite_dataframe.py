'''
dan guess dataframe: word-level dan scores (top 50) 
dan guess processed: a list of (qid, answer, vecs, results)
vw_input format: sentence level features (dan scores + others) 

this script: buzzer requires word level inputs but we don't have word level
features other than dan scores, so to test if the interface between guesser and
buzzer is coded correctly, we rewrite dan guess dataframe into vw_input format
'''
import codecs
from multiprocessing import Pool

from qanta.guesser.abstract import AbstractGuesser
from qanta.util import constants as c

from dqn_buzzer import *

def process_row(row):
    s = "-1 \'{0}_{1}_{2} ".format(row['qnum'], row['sentence'], row['token']) 
    s += "|stats words_seen:{0} ".format(row['token']) # fake feature here
    s += "|guessers {0} DAN_score:{1} DAN_found:1".format(row['guess'], row['score'])
    return s

cfg = config()
for fold in ['dev', 'test']:
    guesses = AbstractGuesser.load_guesses(cfg.guesses_dir, folds=[fold])
    guesses = guesses.to_dict(orient='records')
    rows = Pool(8).map(process_row, guesses)
    outfile = codecs.open(c.VW_INPUT.format(fold), 'w', 'utf-8')
    outfile.write('\n'.join(rows))
