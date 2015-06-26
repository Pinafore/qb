from collections import defaultdict
import argparse
import sqlite3
import sys
import re
import os
from collections import Counter
try:
   import cPickle as pickle
except:
   import pickle

import nltk.classify.util
from nltk.util import ngrams
from unidecode import unidecode
from sklearn.linear_model import LogisticRegression
from nltk.classify.scikitlearn import SklearnClassifier

from util.qdb import QuestionDatabase


alphanum = re.compile('[\W_]+')
merged = {'Biology': 'Science',
          'Physics': 'Science',
          'Chemistry': 'Science',
          'Mathematics': 'Science',
          'Earth Science': 'Science',
          'Astronomy': 'Science',
          'Social Science': 'Social Studies',
          'Geography': 'Social Studies'}

def write_bigrams(bigrams, output):
    o = open(output, 'wb')
    pickle.dump(bigrams, o, pickle.HIGHEST_PROTOCOL)

def classify_text(classifier, text, all_bigrams):
    feats = {}
    total = alphanum.sub(' ', unidecode(text.lower()))
    total = total.split()
    bgs = set(ngrams(total, 2))
    for bg in bgs.intersection(all_bigrams):
        feats[bg] = 1.0
    for word in total:
        feats[word] = 1.0
    return classifier.prob_classify(feats)

def compute_frequent_bigrams(thresh, qbdb):
    # TODO: don't use most frequent bigrams, look by class via feature selection
    # or at least remove stopwords

    bcount = Counter()
    all_questions = qbdb.questions_with_pages()
    train = []
    for page in all_questions:
        for qq in all_questions[page]:
            if qq.fold == 'train':
                for ss, ww, tt in qq.partials():
                    total = ' '.join(tt).strip()
                    total = alphanum.sub(' ', unidecode(total.lower()))
                    total = total.split()
                    bgs = list(ngrams(total, 2))
                    for bg in bgs:
                        bcount[bg] += 1

    return set([k for k, v in bcount.most_common(thresh)])


def train_classifier(out, bgset, questions, attribute, limit=-1):

    all_questions = questions.questions_with_pages()
    c = Counter()
    train = []
    for page in all_questions:
        for qq in all_questions[page]:
            if qq.fold == 'train':
                label = getattr(qq, attribute, "").split(":")[0].lower()
                if not label:
                    continue
                c[label] += 1

                for ss, ww, tt in qq.partials():
                    feats = {}
                    total = ' '.join(tt).strip()
                    total = alphanum.sub(' ', unidecode(total.lower()))
                    total = total.split()

                    # add unigrams
                    for word in total:
                        feats[word] = 1.0

                    # add bigrams
                    currbg = set(ngrams(total, 2))
                    inter = currbg.intersection(bgset)
                    for elem in inter:
                        feats[elem] = 1.0

                    train.append((feats, label))
            if limit > 0 and len(train) > limit:
                break
    print c
    print len(train)
    print("training classifier")
    classifier = SklearnClassifier(LogisticRegression(C=10))
    classifier.train(train)
    pickle.dump(classifier, open(out, 'wb'))
    print 'accuracy@1 train:', nltk.classify.util.accuracy(classifier, train)
    return classifier


def evaluate(classifier_file, bgset, questions, attribute, top=2):
    classifier = pickle.load(open(classifier_file, 'rb'))

    all_questions = questions.questions_with_pages()
    page_num = 0
    c = Counter()
    dev = []
    for page in all_questions:
        page_num += 1
        for qq in all_questions[page]:
            if qq.fold == 'dev':
                label = getattr(qq, attribute, '').split(":")[0].lower()
                if not label:
                    continue
                c[label] += 1

                for ss, ww, tt in qq.partials():
                    feats = {}
                    total = ' '.join(tt).strip()
                    total = alphanum.sub(' ', unidecode(total.lower()))

                    # add unigrams
                    for word in total.split():
                        feats[word] = 1.0

                    # add bigrams
                    currbg = set(ngrams(total, 2))
                    inter = currbg.intersection(bgset)
                    for elem in inter:
                        feats[elem] = 1.0

                    dev.append((feats, label))
    print c
    print len(dev)
    print 'accuracy@1 dev:', nltk.classify.util.accuracy(classifier, dev)
    probs = classifier.prob_classify_many([f for f, a in dev])

    corr = 0.

    for index, prob in enumerate(probs):
        c = Counter()
        for k, v in prob._prob_dict.items():
            c[k] = v
        topn = set([k for k, v in c.most_common(top)])
        if dev[index][1] in topn:
            corr += 1

    print 'top@', top, 'accuracy: ', corr / len(probs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--question_db', type=str, default='data/questions.db')
    parser.add_argument('--attribute', type=str, default='category')
    parser.add_argument('--bigram_thresh', type=int, default=1000)
    parser.add_argument("--output", type=str,
                        default="data/classifier/",
                        help="Where we write output file")

    flags = parser.parse_args()

    questions = QuestionDatabase(flags.question_db)
    bigram_filename = "%s/bigrams.pkl" % flags.output
    if os.path.exists(bigram_filename):
        bgset = pickle.load(open(bigram_filename, 'rb'))
        print("Using previous bigrams")
    else:
        print("computing bigrams...")
        bgset = compute_frequent_bigrams(flags.bigram_thresh, questions)
        write_bigrams(bgset, bigram_filename)

    train_classifier("%s/%s.pkl" % (flags.output, flags.attribute),
                     bgset, questions, flags.attribute)
    evaluate("%s/%s.pkl" % (flags.output, flags.attribute), 
            bgset, questions, flags.attribute)
