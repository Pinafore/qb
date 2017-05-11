import os
import sys
import time
import pickle
import gensim
import numpy as np
import pandas as pd
from collections import namedtuple
from multiprocessing import Pool, Manager
from functools import partial
from typing import List, Dict, Tuple, Optional

from qanta.datasets.quiz_bowl import Question, QuestionDatabase, QuizBowlDataset
from qanta.preprocess import format_guess
from qanta.guesser.abstract import AbstractGuesser
from qanta.util import constants as c
from qanta.util.io import safe_open, safe_path
from qanta.config import conf
from qanta import logging
from qanta.buzzer import constants as bc

log = logging.get(__name__)

GUESSERS = [x.guesser_class for x in AbstractGuesser.list_enabled_guessers()]

def stupid_buzzer(iterator) -> Dict[int, int]:
    '''Buzz by several heuristics.
    '''

    def _do_one(vecs_results_masks):
        vecs, results, masks = vecs_results_masks
        hopeful = any(results == 1)
        prev_top_score = vecs[0][0]
        prev_variance = np.var(vecs[0])
        for i in range(len(masks)):
            if masks[i] == 0:
                return hopeful, results[i - 1], - 1
            if vecs[i][0] - prev_top_score > 0.3:
                if np.var(vecs[i]) - prev_variance > 0.005:
                    return hopeful, results[i], i
        return hopeful, results[-1], - 1

    num_hopeful = 0
    num_results = 0
    tot_reward = 0
    buzz_dict = dict()
    count = 0
    for i in range(iterator.size):
        batch = iterator.next_batch(np)
        count += len(batch.qids)
        vecs = np.swapaxes(batch.vecs, 0, 1)
        results = batch.results.T
        masks = batch.mask.T
        returns = map(_do_one, zip(vecs, results, masks))
        returns = list(map(list, zip(*returns)))
        hopeful = sum(returns[0])
        results = sum(returns[1])
        buzzes = list(returns[2])
        reward = ((results * 10) - (hopeful - results) * 5) / hopeful \
                if hopeful > 0 else 0
        num_hopeful += hopeful
        num_results += results
        tot_reward += reward
        for qid, buzz in zip(batch.qids, buzzes):
            qid = qid.tolist()
            buzz_dict[qid] = buzz
    tot_reward /= iterator.size
    log.info('[stupid] {0} {1} {2}'.format(
        num_hopeful, num_results, tot_reward))
    return buzz_dict

def _process_question(option2id: Dict[str, int], 
        all_questions: List[Question], 
        word2vec, inputs: Tuple) -> \
            Tuple[int, int, List[List[Dict[str, int]]], List[List[int]]]:
    '''Process one question.
    return:
        qnum: question id,
        answer_id: answer id
        guess_dicts: a sequence of guess dictionaries for each guesser
        results: sequence of 0 and 1 for each guesser
    '''
    (qnum, question), queue = inputs

    qnum = int(qnum)
    answer = format_guess(all_questions[qnum].page)
    if answer in option2id:
        answer_id = option2id[answer]
    else:
        answer_id = len(option2id)

    guess_dicts = []
    results = []
    wordvecs = None
    if word2vec is not None:
        wordvecs = []
    for pos, pos_group in question.groupby(['sentence', 'token']):
        pos_group = pos_group.groupby('guesser')
        guess_dicts.append([])
        results.append([])
        if word2vec is not None:
            wordvecs.append([])
        for guesser in GUESSERS:
            if guesser not in pos_group.groups:
                guess_dicts[-1].append({})
                results[-1].append(0)
                if word2vec is not None:
                    wordvecs[-1].append(word2vec.get_zero_vec())
            else:
                guesses = pos_group.get_group(guesser)
                guesses = guesses.sort_values('score', ascending=False)
                top_guess = guesses.iloc[0].guess
                results[-1].append(int(top_guess == answer))
                # normalize score to 0-1 at each time step
                s = sum(guesses.score)
                dic = {x.guess: x.score / s for x in guesses.itertuples()}
                guess_dicts[-1].append(dic)
                if word2vec is not None:
                    wordvecs[-1].append(word2vec.get_avg_vec(top_guess))

    # queue.put(qnum)
    return qnum, answer_id, guess_dicts, results, wordvecs

class Word2Vec:

    def __init__(self, wordvec_dir, wordvec_dim):
        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(
                wordvec_dir, binary=True)
        self.wordvec_dim = wordvec_dim

    def get_avg_vec(self, guess):
        vecs = []
        for word in guess.split('_'):
            if word in self.word2vec:
                vecs.append(self.word2vec[word])
        if len(vecs) > 0:
            return sum(vecs) / len(vecs)
        return self.get_zero_vec()

    def get_zero_vec(self):
        return np.zeros(self.wordvec_dim, dtype=np.float32)


def load_quizbowl(folds=c.BUZZ_FOLDS, use_word2vec=False, multiprocessing=False)\
        -> Tuple[Dict[str, int], Dict[str, list]]:
    log.info('Loading data')
    question_db = QuestionDatabase()
    quizbowl_db = QuizBowlDataset(bc.MIN_ANSWERS)
    all_questions = question_db.all_questions()
    if not os.path.isfile(bc.OPTIONS_DIR):
        log.info('Loading the set of options')
        dev_guesses = AbstractGuesser.load_guesses(bc.GUESSES_DIR, folds=['dev'])
        all_options = set(dev_guesses.guess)

        train_dev_questions = quizbowl_db.questions_in_folds(['train', 'dev'])
        all_options.update({format_guess(q.page) for q in train_dev_questions})

        id2option = list(all_options)
        with open(safe_path(bc.OPTIONS_DIR), 'wb') as outfile:
            pickle.dump(id2option, outfile)
    else:
        with open(safe_path(bc.OPTIONS_DIR), 'rb') as infile:
            id2option = pickle.load(infile)
    option2id = {o: i for i, o in enumerate(id2option)}
    num_options = len(id2option)
    log.info('Number of options {0}'.format(len(id2option)))


    processed_dirs = ['%s_processed.pickle' % (os.path.join(bc.GUESSES_DIR,
        fold)) for fold in folds]
    if not all(os.path.isfile(d) for d in processed_dirs) and use_word2vec:
        log.info('Loading {0}'.format(bc.WORDVEC_DIR))
        word2vec = Word2Vec(bc.WORDVEC_DIR, bc.WORDVEC_DIM)
    else:
        word2vec = None

    guesses_by_fold = dict()
    for k, fold in enumerate(folds):
        save_dir = processed_dirs[k]
        if os.path.isfile(save_dir):
            with open(safe_path(save_dir), 'rb') as infile:
                guesses_by_fold[fold] = pickle.load(infile)
            log.info('Loading {0} guesses'.format(fold))
            continue

        log.info('Processing {0} guesses'.format(fold))
        guesses = AbstractGuesser.load_guesses(bc.GUESSES_DIR, folds=[fold])

        if multiprocessing:
            pool = Pool(conf['buzzer']['n_cores'])
            manager = Manager()
            queue = manager.Queue()
            worker = partial(_process_question, option2id, all_questions, word2vec)
            inputs = [(question, queue) for question in guesses.groupby('qnum')]
            total_size = len(inputs)
            result = pool.map_async(worker, inputs)

            # monitor loop
            while True:
                if result.ready():
                    break
                else:
                    size = queue.qsize()
                    sys.stderr.write('\r[df data] done: {0}/{1}'.format(
                        size, total_size))
                    time.sleep(0.1)
            sys.stderr.write('\n')
            guesses_by_fold[fold] = result.get()

        else:
            returns = []
            guesses = guesses.groupby('qnum')
            total_size = len(guesses)
            for i, question in enumerate(guesses):
                returns.append(_process_question(
                    option2id, all_questions, word2vec, (question, None)))
                sys.stderr.write('\r[df data] done: {0}/{1}'.format(i, total_size))
            guesses_by_fold[fold] = returns

        with open(safe_path(save_dir), 'wb') as outfile:
            pickle.dump(guesses_by_fold[fold], outfile)

        log.info('Processed {0} guesses saved to {1}'.format(fold, save_dir))

    return option2id, guesses_by_fold

def merge_dfs():
    GUESSERS = ["{0}.{1}".format(
        x.guesser_module, x.guesser_class) \
        for x in AbstractGuesser.list_enabled_guessers()]
    log.info("Merging guesser DataFrames.")
    for fold in c.BUZZ_FOLDS:
        new_guesses = pd.DataFrame(columns=['fold', 'guess', 'guesser', 'qnum',
            'score', 'sentence', 'token'], dtype='object')
        for guesser in GUESSERS:
            guesser_dir = os.path.join(c.GUESSER_TARGET_PREFIX, guesser)
            guesses = AbstractGuesser.load_guesses(guesser_dir, folds=[fold])
            new_guesses = new_guesses.append(guesses)
        for col in ['qnum', 'sentence', 'token', 'score']:
            new_guesses[col] = pd.to_numeric(new_guesses[col], downcast='integer')
        merged_dir = os.path.join(c.GUESSER_TARGET_PREFIX, 'merged')
        if not os.path.exists(merged_dir):
            os.makedirs(merged_dir)
        AbstractGuesser.save_guesses(new_guesses, merged_dir, folds=[fold])
        log.info("Merging: {0} finished.".format(fold))

if __name__ == "__main__":
    merge_dfs()
    option2id, guesses_by_fold = load_quizbowl(c.BUZZ_FOLDS, use_word2vec=True)
