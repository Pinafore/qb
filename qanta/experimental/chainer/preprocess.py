import re
import os
import json
import spacy
from tqdm import tqdm
import logging

from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.util.constants import GUESSER_TRAIN_FOLD, GUESSER_DEV_FOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load('en')

UNK = '<UNK>'
EOS = '<EOS>'
UNK_ID = 0
EOS_ID = 1
LOWER = True
VOCAB_SIZE = 5

answers_dir = 'answers.json'
vocabulary_dir = 'vocabulary.json'
train_questions_dir = 'train_questions.json'
dev_questions_dir = 'dev_questions.json'

db = QuestionDatabase()
all_questions = list(db.all_questions().values())

if os.path.isfile(answers_dir):
    logger.info('loading answers')
    with open(answers_dir, 'r') as f:
        checkpoint = json.loads(f.read())
        id_to_answer = checkpoint['id_to_answer']
        answer_to_id = checkpoint['answer_to_id']
else:
    logger.info('processing answers..', end='')
    id_to_answer = list(set(x.page for x in all_questions))
    answer_to_id = {x: i for i, x in enumerate(id_to_answer)}
    with open('answers.json', 'w') as f:
        answers = {'id_to_answer': id_to_answer, 'answer_to_id': answer_to_id}
        f.write(json.dumps(answers))
logger.info('number of answers: {}'.format(len(id_to_answer)))

def clean_question(text):
    return re.sub('\s+', ' ',
            re.sub(r'[~\*\(\)]|--', ' ', text)
            ).strip()

vocabulary_updated = False
if os.path.isfile(vocabulary_dir):
    logger.info('loading vocabulary')
    with open(vocabulary_dir, 'r') as f:
        checkpoint = json.loads(f.read())
        id_to_word = checkpoint['id_to_word']
        word_to_id = checkpoint['word_to_id']
else:
    logger.info('processing vocabulary..')
    id_to_word = [UNK, EOS]
    word_to_id = {UNK: UNK_ID, EOS: EOS_ID}
    
    # create vocabulary from all the data
    # also replace question.text with spacy doc
    for i, question in enumerate(tqdm(all_questions)):
        text = nlp(clean_question(question.flatten_text()))
        all_questions[i].text = text
        for word in text:
            if word.is_alpha or word.is_digit:
                word = word.lower_ if LOWER else word.text
                if word not in word_to_id:
                    word_to_id[word] = len(id_to_word)
                    id_to_word.append(word)

    with open(vocabulary_dir, 'w') as f:
        vocabulary = {'id_to_word': id_to_word, 'word_to_id': word_to_id}
        f.write(json.dumps(vocabulary))
    vocabulary_updated = True
logger.info('vocabulary size: {}'.format(len(id_to_word)))

# convert train and dev data
if os.path.isfile(train_questions_dir) \
        and os.path.isfile(dev_questions_dir) \
        and not vocabulary_updated:
    logger.info('loading questions')
    with open(train_questions_dir, 'r') as f:
        train_questions = json.loads(f.read())
    with open(dev_questions_dir, 'r') as f:
        dev_questions = json.loads(f.read())
else:
    logger.info('processing questions')
    train_questions = dict()
    dev_questions = dict()
    for question in tqdm(all_questions):
        if question.fold not in [GUESSER_TRAIN_FOLD, GUESSER_DEV_FOLD]:
            continue
    
        if isinstance(question.text, dict):
            question.text = nlp(clean_question(question.flatten_text()))
    
        sents = []
        for sent in question.text.sents:
            _sent = []
            for word in sent:
                word = word.lower_ if LOWER else word.text
                word = word_to_id.get(word, UNK_ID)
                _sent.append(word)
            sents.append(_sent)
    
        q = {'sentences': sents, 'answer': question.page}
        if question.fold == GUESSER_TRAIN_FOLD:
            train_questions[question.qnum] = q
        elif question.fold == GUESSER_DEV_FOLD:
            dev_questions[question.qnum] = q

    with open('train_questions.json', 'w') as f:
        f.write(json.dumps(train_questions))
    
    with open('dev_questions.json', 'w') as f:
        f.write(json.dumps(dev_questions))

logger.info('number of training questions: {}'.format(len(train_questions)))
logger.info('number of dev questions: {}'.format(len(dev_questions)))
