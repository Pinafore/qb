import os
import sys
import json
import time
import pickle
import codecs
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
from qanta.util.multiprocess import _multiprocess

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
        all_questions: Dict[int, Question], 
        word2vec, qnum, question) -> \
            Tuple[int, int, List[List[Dict[str, int]]], List[List[int]]]:
    '''Process one question.
    return:
        qnum: question id,
        answer_id: answer id
        guess_dicts: a sequence of guess dictionaries for each guesser
        results: sequence of 0 and 1 for each guesser
    '''
    qnum = int(qnum)
    try:
        answer = all_questions[qnum].page
    except KeyError:
        return None
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
                log.info("{0} missing guesser {1}.".format(qnum, guesser))
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


def load_quizbowl(folds=c.BUZZER_INPUT_FOLDS, word2vec=None) -> Tuple[Dict[str, int], Dict[str, list]]:
    merge_dfs()
    log.info('Loading data')
    question_db = QuestionDatabase()
    quizbowl_db = QuizBowlDataset(bc.MIN_ANSWERS, guesser_train=True, buzzer_train=True)
    all_questions = question_db.all_questions()
    if not os.path.isfile(bc.OPTIONS_DIR):
        log.info('Loading the set of options')
        all_options = set(quizbowl_db.training_data()[1])

        id2option = list(all_options)
        with open(safe_path(bc.OPTIONS_DIR), 'wb') as outfile:
            pickle.dump(id2option, outfile)
    else:
        with open(safe_path(bc.OPTIONS_DIR), 'rb') as infile:
            id2option = pickle.load(infile)
    option2id = {o: i for i, o in enumerate(id2option)}
    num_options = len(id2option)
    log.info('Number of options {0}'.format(len(id2option)))

    guesses_by_fold = dict()
    for fold in folds:
        save_dir = '%s_processed.pickle' % (os.path.join(bc.GUESSES_DIR, fold))
        if os.path.isfile(save_dir):
            with open(safe_path(save_dir), 'rb') as infile:
                guesses_by_fold[fold] = pickle.load(infile)
            log.info('Loading {0} guesses'.format(fold))
            continue

        log.info('Processing {0} guesses'.format(fold))
        guesses = AbstractGuesser.load_guesses(bc.GUESSES_DIR, folds=[fold])

        worker = partial(_process_question, option2id, all_questions, word2vec)
        inputs = guesses.groupby('qnum')
        guesses_by_fold[fold] = _multiprocess(worker, inputs, info='df data',
                multi=True)
        guesses_by_fold[fold] = [x for x in guesses_by_fold[fold] if x is not None]

        with open(safe_path(save_dir), 'wb') as outfile:
            pickle.dump(guesses_by_fold[fold], outfile)

        log.info('Processed {0} guesses saved to {1}'.format(fold, save_dir))

    return option2id, guesses_by_fold

def merge_dfs():
    GUESSERS = ["{0}.{1}".format(
        x.guesser_module, x.guesser_class) \
        for x in AbstractGuesser.list_enabled_guessers()]
    log.info("Merging guesser DataFrames.")
    merged_dir = os.path.join(c.GUESSER_TARGET_PREFIX, 'merged')
    if not os.path.exists(merged_dir):
        os.makedirs(merged_dir)
    for fold in c.BUZZER_INPUT_FOLDS:
        if os.path.exists(AbstractGuesser.guess_path(merged_dir, fold)):
            log.info("Merged {0} exists, skipping.".format(fold))
            continue
        new_guesses = pd.DataFrame(columns=['fold', 'guess', 'guesser', 'qnum',
            'score', 'sentence', 'token'], dtype='object')
        for guesser in GUESSERS:
            guesser_dir = os.path.join(c.GUESSER_TARGET_PREFIX, guesser)
            guesses = AbstractGuesser.load_guesses(guesser_dir, folds=[fold])
            new_guesses = new_guesses.append(guesses)
        for col in ['qnum', 'sentence', 'token', 'score']:
            new_guesses[col] = pd.to_numeric(new_guesses[col], downcast='integer')
        AbstractGuesser.save_guesses(new_guesses, merged_dir, folds=[fold])
        log.info("Merging: {0} finished.".format(fold))

def load_protobowl():
    protobowl_df_dir = bc.PROTOBOWL_DIR + '.df.pkl'
    if os.path.exists(protobowl_df_dir):
        with open(protobowl_df_dir, 'rb') as f:
            return pickle.load(f)
    
    def process_line(x):
        total_time = x['object']['time_elapsed'] + x['object']['time_remaining']
        ratio = x['object']['time_elapsed'] / total_time
        position = int(len(x['object']['question_text'].split()) * ratio)
        return [x['object']['guess'], x['object']['qid'], 
                position, x['object']['ruling']]

    data = []
    count = 0
    with codecs.open(bc.PROTOBOWL_DIR, 'r', 'utf-8') as f:
        line = f.readline()
        while line is not None:
            line = line.strip()
            if len(line) < 1:
                break
            while not line.endswith('}}'):
                l = f.readline()
                if l is None:
                    break
                line += l.strip()
            try:
                line = json.loads(line)
            except ValueError:
                line = f.readline()
                if line == None:
                    break
                continue
            count += 1
            if count % 10000 == 0:
                sys.stderr.write('\rdone: {}'.format(count))
            data.append(process_line(line))
            line = f.readline()

    protobowl_df = df = pd.DataFrame(data, 
            columns=['guess', 'qid', 'position', 'result'])
    with open(protobowl_df_dir, 'wb') as f:
        pickle.dump(protobowl_df, f)

    return protobowl_df

if __name__ == "__main__":
    merge_dfs()

    processed_dirs = ['%s_processed.pickle' % (os.path.join(
        bc.GUESSES_DIR, fold)) for fold in c.BUZZER_INPUT_FOLDS]
    # if not all(os.path.isfile(d) for d in processed_dirs):
    #     log.info('Loading {0}'.format(bc.WORDVEC_DIR))
    #     word2vec = Word2Vec(bc.WORDVEC_DIR, bc.WORDVEC_DIM)
    # else:
    word2vec = None

    option2id, guesses_by_fold = load_quizbowl(c.BUZZER_INPUT_FOLDS, word2vec)
    load_protobowl()
