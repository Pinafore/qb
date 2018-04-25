import re
import json
import os
import spacy
import collections
from tqdm import tqdm

from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.util.constants import GUESSER_TRAIN_FOLD, GUESSER_DEV_FOLD

from nlp_utils import make_vocab
from nlp_utils import transform_to_array

def clean_question(text):
    return re.sub('\s+', ' ',
            re.sub(r'[~\*\(\)]|--', ' ', text)
            ).strip()

def load_quizbowl(split_sentences=True, num_answers=-1, min_answer_freq=-1):
    nlp = spacy.load('en')
    questions = QuestionDatabase().all_questions().values()
    answers = [x.page for x in questions]
    answer_counter = collections.Counter(answers)
    if num_answers != -1:
        answer_counter = sorted(answer_counter.items(), key=lambda x: x[1])[::-1]
        answers = [x for x, y in answer_counter[:num_answers]]
    else:
        answers = [x for x, y in answer_counter.items() if y >= min_answer_freq]
    answer_to_id = {x: i for i, x in enumerate(answers)}
    print('# class: {}'.format(len(answers)))

    folds = [GUESSER_TRAIN_FOLD, GUESSER_DEV_FOLD]
    questions = [x for x in questions if x.fold in folds \
            and x.page in answers]

    train, dev = [], []
    for q in tqdm(questions):
        text = nlp(clean_question(q.flatten_text()))
        answer = answer_to_id[q.page]
        if split_sentences:
            for sent in text.sents:
                sent = [w.lower_ for w in sent if w.is_alpha or w.is_digit]
                if q.fold == GUESSER_TRAIN_FOLD:
                    train.append((sent, answer))
                else:
                    dev.append((sent, answer))
        else:
            sent = [w.lower_ for w in text if w.is_alpha or w.is_digit]
            if q.fold == GUESSER_TRAIN_FOLD:
                train.append((sent, answer))
            else:
                dev.append((sent, answer))

    return train, dev, answers

def get_quizbowl(data_dir='data/nn_guesser', split_sentences=True,
        num_answers=-1, min_answer_freq=-1):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    train_dir = os.path.join(data_dir, 'train.json')
    dev_dir = os.path.join(data_dir, 'dev.json')
    answers_dir = os.path.join(data_dir, 'answers.json')
    existance = [os.path.isfile(x) for x in [train_dir, dev_dir, answers_dir]]

    if all(existance):
        with open(train_dir, 'r') as f:
            train = json.loads(f.read())
        with open(dev_dir, 'r') as f:
            dev = json.loads(f.read())
        with open(answers_dir, 'r') as f:
            answers = json.loads(f.read())
    else:
        train, dev, answers = load_quizbowl(
                split_sentences, num_answers, min_answer_freq)
        with open(train_dir, 'w') as f:
            f.write(json.dumps(train))
        with open(dev_dir, 'w') as f:
            f.write(json.dumps(dev))
        with open(answers_dir, 'w') as f:
            f.write(json.dumps(answers))

    print('# train data: {}'.format(len(train)))
    print('# dev data: {}'.format(len(dev)))
    print('# class: {}'.format(len(answers)))

    vocab_dir = os.path.join(data_dir, 'vocab.json')
    if os.path.isfile(vocab_dir):
        with open(vocab_dir, 'r') as f:
            vocab = json.loads(f.read())
    else:
        vocab = make_vocab(train)
        with open(vocab_dir, 'w') as f:
            f.write(json.dumps(vocab))

    print('# vocab: {}'.format(len(vocab)))

    train = transform_to_array(train, vocab)
    dev = transform_to_array(dev, vocab)

    return train, dev, vocab, answers

if __name__ == '__main__':
    train, dev, vocab, answers = get_quizbowl()
