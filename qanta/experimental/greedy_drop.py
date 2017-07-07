import math
import pickle
from collections import defaultdict
from typing import List, Dict

from qanta.config import conf
from qanta.util.io import safe_path
from qanta.util.multiprocess import _multiprocess
from qanta.guesser.abstract import AbstractGuesser
from qanta.datasets.quiz_bowl import QuizBowlDataset, Question
from qanta.guesser.experimental.elasticsearch_instance_of import ElasticSearchWikidataGuesser


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
        guesses: the new guesses.
    '''
    question = ' '.join(question[:i] + question[i+1:])
    guesses = guesser.guess_single(question)
    return guesses

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
    dict1 = guesser.guess_single(' '.join(question))
    for i in range(n):
        inputs = [' '.join(question[:j] + question[j+1:]) for j in range(len(question))]
        dict2s = map(guesser.guess_single, inputs)
        scores = [kl(dict1, dict2) for dict2 in dict2s]
        bext = sorted(list(enumerate(scores)), key=lambda x: x[1])[0][0]
        dropped.append(indices[bext])
        question = question[:bext] + question[bext + 1:]
        indices = indices[:bext] + indices[bext + 1:]
    return ' '.join(question), dropped

def evaluate_question(question: Question, first_n=-1, keep_n=lambda x: 6):
    '''Get the top one and top five precisions before and after dropping words.
    Args:
        question: a Question object.
        first_n: take the first_n sentences.
        keep_n: number of words to keep as a function of total number.
    Return:
        text_before: question text after dropping.
        text_after: question text after dropping.
        guesses_before: question text after dropping.
        guesses_after: question text after dropping.
        dropped: indices of words that are dropped.
    '''
    sents = list(question.text.values())
    if first_n == -1:
        first_n = len(sents)
    text_before = ' '.join(sents[:first_n])
    text_after, dropped = greedy_drop(text_before, keep_n)
    guesses_before = guesser.guess_single(text_before).items()
    guesses_after = guesser.guess_single(text_after).items()
    guesses_before = sorted(guesses_before, key=lambda x: x[1], reverse=True)
    guesses_after = sorted(guesses_after, key=lambda x: x[1], reverse=True)
    return text_before, text_after, guesses_before, guesses_after, dropped

def run(questions, first_n, keep_n, ckp_dir):
    inputs = [[x, first_n, keep_n] for x in questions]
    returns = _multiprocess(evaluate_question, inputs)
    returns = list(map(list, zip(*returns)))
    text_before, text_after, guesses_before, guesses_after, dropped = returns
    checkpoint = defaultdict(dict)
    s = [[], [], [], []]
    for i, q in enumerate(questions):
        checkpoint[q.qnum]['text_before'] = text_before[i]
        checkpoint[q.qnum]['text_after'] = text_after[i]
        checkpoint[q.qnum]['guesses_before'] = guesses_before[i]
        checkpoint[q.qnum]['guesses_after'] = guesses_after[i]
        checkpoint[q.qnum]['dropped'] = dropped[i]
        gb = guesses_before[i]
        ga = guesses_after[i]
        s[0].append(gb[0][0] == q.page)
        s[1].append(gb[0][0] == q.page)
        s[2].append(q.page in [x[0] for x in gb[:5]])
        s[3].append(q.page in [x[0] for x in ga[:5]])
    print([sum(x) / len(x) for x in s])

    with open(safe_path(ckp_dir), 'wb') as f:
        pickle.dump(checkpoint, f)

def quarter(x):
    return min(x // 4, 5)

def five(x):
    return 5

def main():
    fold = 'guessdev'
    db = QuizBowlDataset(1, guesser_train=True, buzzer_train=True)
    questions = db.questions_in_folds([fold])
    ckp_dir = 'output/experimental/{0}.{1}.first.pkl'.format(fold, 'quarter')
    run(questions, 1, quarter, ckp_dir)

if __name__ == '__main__':
    main()
