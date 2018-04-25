import math
import pickle
from collections import defaultdict
from typing import List, Dict

from qanta.config import conf
from qanta.util.io import safe_path
from qanta.util.multiprocess import _multiprocess
from qanta.guesser.abstract import AbstractGuesser
from qanta.datasets.quiz_bowl import QuizBowlDataset, Question
from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.guesser.experimental.elasticsearch_instance_of import ElasticSearchWikidataGuesser


'''Greedily drop words while trying to maintain the guesses
Difference between two sets of guesses is measured by kl divergence
'''

gspec = AbstractGuesser.list_enabled_guessers()[0]
guesser_dir = AbstractGuesser.output_path(gspec.guesser_module,
        gspec.guesser_class, '')
guesser = ElasticSearchWikidataGuesser.load(guesser_dir)


def kl(dict1: Dict[str, float], dict2: Dict[str, float]) -> float:
    '''Normalize the two dictionaries and compute the KL divergence KL(d1 | d2)
    '''
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

def greedy_remove(text_before, guesses_before, n_keep):
    '''Remove words from the question while trying to keep the original
        predictions
    Args:
        text_before: the text before removal
        guesses_before: a dictionary of scores of guesses as the starting point
        n_keep: number of words to keep, either a constant or a function of the
                total number of words
    Retuen:
        text_after
        guesses_after
        removed: the original indices of words that are removed
    '''
    text = text_before.split()

    # calculate the number of words to remove
    if callable(n_keep):
        n_keep = n_keep(len(text))
    # number of words to keep cannot exceed total number of words
    n_keep = min(len(text), n_keep)
    n_remove = len(text) - n_keep

    removed = []
    indices = list(range(len(text)))
    dict1 = guesses_before
    for i in range(n_remove):
        # iterate through all possible words to remove at this step
        inputs = [' '.join(text[:j] + text[j+1:]) 
                for j in range(len(text))]
        dict2s = map(guesser.guess_single, inputs)
        scores = [kl(dict1, dict2) for dict2 in dict2s]
        idx = sorted(list(enumerate(scores)), key=lambda x: x[1])[0][0]
        removed.append(indices[idx])
        text = text[:idx] + text[idx + 1:]
        indices = indices[:idx] + indices[idx + 1:]

    text_after = ' '.join(text)
    guesses_after = guesser.guess_single(text_after)
    return text_after, guesses_after, removed

def main(questions, n_keep, ckp_dir):
    db = QuizBowlDataset(guesser_train=True, buzzer_train=True)
    questions = db.questions_in_folds(['guessdev'])
    questions = {x.qnum: x for x in questions}

    checkpoint = defaultdict(dict)
    for qnum, question in questions.items():
        text_before = question.flatten_text()
        guesses_before = guesser.guess_single(text_before)
        text_after, guesses_after, removed = greedy_remove(
                text_before, guesses_before, n_keep)
        checkpoint[qnum]['text_before'] = text_before
        checkpoint[qnum]['text_after'] = text_after
        checkpoint[qnum]['guesses_before'] = guesses_before
        checkpoint[qnum]['guesses_after'] = guesses_after
        checkpoint[qnum]['removed'] = removed

    checkpoint = dict(checkpoint)
    with open(safe_path(ckp_dir), 'wb') as f:
        pickle.dump(checkpoint, f)

    evaluate(ckp_dir)

def evaluate(ckp_dir):
    db = QuizBowlDataset(guesser_train=True, buzzer_train=True)
    questions = db.questions_in_folds(['guessdev'])
    questions = {x.qnum: x for x in questions}

    with open(ckp_dir, 'rb') as f:
        checkpoint = pickle.load(f)

    scores = [0, 0, 0, 0, 0]
    descriptions = ['accuracy before', 'accuracy after', 'before after match',
                    'top 5 accuracy before', 'top 5 accuracy after']
    for k, q in checkpoint.items():
        page = questions[k].page
        gb = sorted(q['guesses_before'].items(), key=lambda x: x[1])[::-1]
        ga = sorted(q['guesses_after'].items(), key=lambda x: x[1])[::-1]
        scores[0] += gb[0][0] == page # accuracy before
        scores[1] += ga[0][0] == page # accuracy after
        scores[2] += ga[0][0] == gb[0][0] # top 1 match before / after
        scores[3] += page in [x[0] for x in gb[:5]] # top 5 accuracy before
        scores[4] += page in [x[0] for x in ga[:5]] # top 5 accuracy after
    scores = [x / len(questions) for x in scores]
    for s, d in zip(scores, descriptions):
        print(d, s)

if __name__ == '__main__':
    n_keep = 20
    ckp_dir = 'output/experimental/greedy_remove.dev.{}.pkl'.format(n_keep)
    # evaluate(ckp_dir)
    # main(n_keep, ckp_dir)
