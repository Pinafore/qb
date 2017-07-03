import math
from functools import partial
from elasticsearch_dsl.connections import connections
from qanta.guesser.abstract import AbstractGuesser
from qanta.guesser.experimental.elasticsearch_instance_of import ElasticSearchWikidataGuesser
from qanta.datasets.quiz_bowl import QuizBowlDataset

'''Greedily drop words while trying to maintain the guesses.
Difference between two sets of guesses is measured by kl divergence.'''

def kl(dict1, dict2):
    def normalize(d):
        s = sum(d.values())
        for k, v in d.items():
            d[k] = v / s
        return d

    x1 = normalize(dict1)
    x2 = normalize(dict2)
    score = 0
    for k in x1.keys():
        score += x1[k] * math.log(x1[k] / x2.get(k, x1[k] / 2))
    return score

def drop(question, dict1, i):
    '''Get the divergence between dict1 and question with ith word dropped'''
    if isinstance(question, str):
        question = question.split()
    question = ' '.join(question[:i] + question[i+1:])
    dict2 = dict(guesser.guess_single(question))
    return kl(dict1, dict2)

def greedy_drop(question, n):
    '''Drop n words from the question.'''
    if isinstance(question, str):
        question = question.split()
    assert n < len(question)
    dropped = []
    indices = list(range(len(question)))
    dict1 = dict(guesser.guess_single(' '.join(question)))
    for i in range(n):
        worker = partial(drop, question, dict1)
        scores = [worker(j) for j in range(len(question))]
        bext = sorted(list(enumerate(scores)), key=lambda x: x[1])[0][0]
        dropped.append(indices[bext])
        question = question[:bext] + question[bext + 1:]
        indices = indices[:bext] + indices[bext + 1:]
    return question, dropped

connections.create_connection(hosts=['localhost'])
gspec = AbstractGuesser.list_enabled_guessers()[0]
guesser_dir = AbstractGuesser.output_path(gspec.guesser_module,
        gspec.guesser_class, '')
guesser = ElasticSearchWikidataGuesser.load(guesser_dir)
db = QuizBowlDataset(1, guesser_train=True, buzzer_train=True)
questions = db.questions_in_folds(['guessdev'])

before = ' '.join(list(questions[20].text.values())).split()
after, dropped = greedy_drop(before, 50)
