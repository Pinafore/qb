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

def greedy_remove(question: Question, n_keep, n_sents):
    '''Remove words from the question while trying to keep the original
        predictions
    Args:
        question: the question to be modified
        n_sents: take the first n_sents sentences, either a constant or a
                function of the total number of sentences
        n_keep: number of words to keep, either a constant or a function of the
                total number of words
    Retuen:
        text_before
        text_after
        guesses_before
        guesses_after
        removed: the original indices of words that are removed
    '''
    # take the first n_sents sentences
    sents = list(question.text.values())
    if callable(n_sents):
        n_sents = n_sents(len(sents))
    text_before = ' '.join(sents[:n_sents])
    guesses_before = guesser.guess_single(text_before)
    question = text_before.split()

    # calculate the number of words to remove
    if callable(n_keep):
        n_keep = n_keep(len(question))
    # number of words to keep cannot exceed total number of words
    n_keep = min(len(question), n_keep)
    n_remove = len(question) - n_keep

    removed = []
    indices = list(range(len(question)))
    dict1 = guesses_before
    for i in range(n_remove):
        # iterate through all possible words to remove at this step
        inputs = [' '.join(question[:j] + question[j+1:]) 
                for j in range(len(question))]
        dict2s = map(guesser.guess_single, inputs)
        scores = [kl(dict1, dict2) for dict2 in dict2s]
        idx = sorted(list(enumerate(scores)), key=lambda x: x[1])[0][0]
        removed.append(indices[idx])
        question = question[:idx] + question[idx + 1:]
        indices = indices[:idx] + indices[idx + 1:]

    text_after = ' '.join(question)
    guesses_after = guesser.guess_single(text_after)
    return text_before, text_after, guesses_before, guesses_after, removed

def all_sents(x):
    # take all sentences from the question
    return x

def main():
    fold = 'guessdev'
    db = QuizBowlDataset(guesser_train=True, buzzer_train=True)
    questions = db.questions_in_folds([fold])

    n_sents = all_sents 
    n_keep = 8 # keep 8 words

    inputs = [[x, n_keep, n_sents] for x in questions]
    returns = _multiprocess(greedy_remove, inputs)
    returns = list(map(list, zip(*returns)))
    text_before, text_after, guesses_before, guesses_after, removed = returns

    checkpoint = defaultdict(dict)
    scores = [0, 0, 0, 0, 0]
    descriptions = ['accuracy before', 'accuracy after', 'before after match',
                    'top 5 accuracy before', 'top 5 accuracy after']
    for i, q in enumerate(questions):
        checkpoint[q.qnum]['text_before'] = text_before[i]
        checkpoint[q.qnum]['text_after'] = text_after[i]
        checkpoint[q.qnum]['guesses_before'] = guesses_before[i]
        checkpoint[q.qnum]['guesses_after'] = guesses_after[i]
        checkpoint[q.qnum]['removed'] = removed[i]
        gb = sorted(guesses_before[i].items(), key=lambda x: x[1])[::-1]
        ga = sorted(guesses_after[i].items(), key=lambda x: x[1])[::-1]
        scores[0] += gb[0][0] == q.page # accuracy before
        scores[1] += ga[0][0] == q.page # accuracy after
        scores[2] += ga[0][0] == gb[0][0] # top 1 match before / after
        scores[3] += q.page in [x[0] for x in gb[:5]] # top 5 accuracy before
        scores[4] += q.page in [x[0] for x in ga[:5]] # top 5 accuracy after
    scores = [x / len(questions) for x in scores]
    for s, d in zip(scores, descriptions):
        print(d, s)

    ckp_dir = 'output/experimental/greedy_remove.{0}.{1}.pkl'.format(fold, '8')

    checkpoint = dict(checkpoint)
    with open(safe_path(ckp_dir), 'wb') as f:
        pickle.dump(checkpoint, f)

if __name__ == '__main__':
    main()
