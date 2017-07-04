import math
import pickle
from functools import partial
from typing import List, Dict, Tuple, Optional

from qanta.guesser.abstract import AbstractGuesser
from qanta.guesser.experimental.elasticsearch_instance_of import ElasticSearchWikidataGuesser
from qanta.datasets.quiz_bowl import QuizBowlDataset, Question
from qanta.util.io import safe_path
from qanta.util.multiprocess import _multiprocess


'''Greedily drop words while trying to maintain the guesses.
Difference between two sets of guesses is measured by kl divergence.'''


gspec = AbstractGuesser.list_enabled_guessers()[0]
guesser_dir = AbstractGuesser.output_path(gspec.guesser_module,
        gspec.guesser_class, '')
guesser = ElasticSearchWikidataGuesser.load(guesser_dir)

def kl(dict1: Dict[str, float], dict2: Dict[str, float]) -> float:
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

def drop(question: List[str], dict1: Dict[str, float], i: int) -> float:
    '''Get the divergence between dict1 and question with ith word dropped.
    Args:
        question: the unmodified question.
        dict1: the guesses from the unmodified question.
        i: the position of the word to be dropped.
    Return:
        the KL divergence between the guesses from modified and original
        question.
    '''
    question = ' '.join(question[:i] + question[i+1:])
    dict2 = dict(guesser.guess_single(question))
    return kl(dict1, dict2)

def greedy_drop(question, keep_n):
    '''Drop n words from the question.
    Args:
        question: the question to be modified.
        keep_n: number of words to keep as a function of the total number.
    Retuen:
        question: the question after dropping n words.
        dropped: the indices of words that are dropped.
    '''
    if isinstance(question, str):
        question = question.split()
    if callable(keep_n):
        keep_n = keep_n(len(question))
    keep_n = min(len(question), keep_n)
    n = len(question) - keep_n
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
    return ' '.join(question), dropped

def evaluate_question(question: Question, first_n=-1, keep_n=lambda: 6):
    '''Get the top one and top five precisions before and after dropping words.
    Args:
        question: a Question object.
        first_n: take the first_n sentences.
        keep_n: number of words to keep as a function of total number.
    Return:
        text_before: question text after dropping.
        text_after: question text after dropping.
        dropped: indices of words that are dropped.
        top_one_before: top one precision before dropping.
        top_one_after: top one precision after dropping.
        top_five_before: top five precision before dropping.
        top_five_after: top five precision after dropping.
    '''
    sents = list(question.text.values())
    if first_n == -1:
        first_n = len(sents)
    text_before = ' '.join(sents[:first_n])
    text_after, dropped = greedy_drop(text_before, keep_n)
    guesses_before = guesser.guess_single(text_before)
    guesses_after = guesser.guess_single(text_after)
    guesses_before = sorted(guesses_before, key=lambda x: x[1], reverse=True)
    guesses_after = sorted(guesses_before, key=lambda x: x[1], reverse=True)
    guesses_before = [x[0] for x in guesses_before]
    guesses_after = [x[0] for x in guesses_after]
    t1b = (guesses_before[0] == question.page)
    t1f = (guesses_after[0] == question.page)
    t5b = (question.page in guesses_before[:5])
    t5f = (question.page in guesses_after[:5])
    return text_before, text_after, dropped, t1b, t1f, t5b, t5f

def main():
    fold = 'guessdev'
    db = QuizBowlDataset(1, guesser_train=True, buzzer_train=True)
    questions = db.questions_in_folds([fold])[:10]

    inputs = [[x, -1, 6] for x in questions]
    returns = _multiprocess(evaluate_question, inputs, multi=True)
    returns = list(map(list, zip(*returns)))
    text_before, text_after, dropped, t1b, t1f, t5b, t5f = returns
    text_after = {q.qnum: t for q, t in zip(questions, text_after)}
    dropped = {q.qnum: d for q, d in zip(questions, dropped)}
    print([sum(x) / len(x) for x in [t1b, t1f, t5b, t5f]])

    pkl_dir = 'output/experimental/{}'.format(fold)
    with open(safe_path(pkl_dir + 'after.all_sents.pkl'), 'wb') as f:
        pickle.dump(text_after, f)
    with open(safe_path(pkl_dir + 'dropped.all_sents.pkl'), 'wb') as f:
        pickle.dump(dropped, f)

    inputs = [[x, 1, 6] for x in questions]
    returns = _multiprocess(evaluate_question, inputs, multi=True)
    returns = list(map(list, zip(*returns)))
    text_before, text_after, dropped, t1b, t1f, t5b, t5f = returns
    text_after = {q.qnum: t for q, t in zip(questions, text_after)}
    dropped = {q.qnum: d for q, d in zip(questions, dropped)}
    print([sum(x) / len(x) for x in [t1b, t1f, t5b, t5f]])

    pkl_dir = 'output/experimental/{}'.format(fold)
    with open(safe_path(pkl_dir + 'after.first_sents.pkl'), 'wb') as f:
        pickle.dump(text_after, f)
    with open(safe_path(pkl_dir + 'dropped.first_sents.pkl'), 'wb') as f:
        pickle.dump(dropped, f)

if __name__ == '__main__':
    main()
