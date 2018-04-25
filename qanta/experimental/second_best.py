import re
import math
import pickle
import itertools
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict

from elasticsearch_dsl import Search
from elasticsearch_dsl.connections import connections

from qanta.config import conf
from qanta.util.io import safe_path
from qanta.util.multiprocess import _multiprocess
from qanta.guesser.abstract import AbstractGuesser
from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.guesser.experimental.elasticsearch_instance_of import ElasticSearchWikidataGuesser

gspec = AbstractGuesser.list_enabled_guessers()[0]
guesser_dir = AbstractGuesser.output_path(gspec.guesser_module,
        gspec.guesser_class, '')
guesser = ElasticSearchWikidataGuesser.load(guesser_dir)

def get_second_best_wiki_words(question):
    text = question.flatten_text()
    # query top 10 guesses
    s = Search(index='qb_ir_instance_of')[0:10].query('multi_match', query=text,
            fields=['wiki_content', 'qb_content', 'source_content'])
    s = s.highlight('qb_content').highlight('wiki_content')
    results = list(s.execute())
    guess = results[1] # take the second best answer
    _highlights = guess.meta.highlight 

    try:
        wiki_content = list(_highlights.wiki_content)
    except AttributeError:
        wiki_content = None

    try:
        qb_content = list(_highlights.qb_content)
    except AttributeError:
        qb_content = None

    words = {}
    if wiki_content is None:
        words['wiki'] = None
    else:
        words['wiki'] = itertools.chain(*[re.findall('<em>(.*?)</em>', x) for x in list(wiki_content)])

    if qb_content is None:
        words['qb'] = None
    else:
        words['qb'] = itertools.chain(*[re.findall('<em>(.*?)</em>', x) for x in list(qb_content)])

    return words

def test():
    questions = QuestionDatabase().all_questions()
    guessdev_questions = [v  for k, v in questions.items() 
            if v.fold == 'guessdev']
    q = guessdev_questions[1]
    second_best_words = get_second_best_wiki_words(q)
    guesses = guesser.guess_single(q.flatten_text())
    guesses = sorted(guesses.items(), key=lambda x: x[1])
    guess_before = guesses[-1]
    print(q.flatten_text())
    print(guess_before)
    print()

    text_after = q.flatten_text()
    if second_best_words['wiki'] is not None: 
        text_after += ' '.join(second_best_words['wiki'])
    if second_best_words['qb'] is not None: 
        text_after += ' '.join(second_best_words['qb'])
    guesses = guesser.guess_single(text_after)
    guesses = sorted(guesses.items(), key=lambda x: x[1])
    guess_after = guesses[-1]
    print(text_after)
    print(guess_after)


def main():
    questions = QuestionDatabase().all_questions()
    guessdev_questions = {k: v  for k, v in questions.items() 
            if v.fold == 'guessdev'}

if __name__ == '__main__':
    test()
