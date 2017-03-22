import pickle

import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from qanta import logging
from qanta.util.io import safe_open, safe_path
from qanta.util.constants import (CLASSIFIER_PICKLE_PATH, CLASSIFIER_REPORT_PATH,
                                  CLASSIFIER_GUESS_PROPS)
from qanta.util.environment import QB_QUESTION_DB
from qanta.preprocess import format_guess
from qanta.datasets.quiz_bowl import QuestionDatabase, QuizBowlDataset
from qanta.reporting.report_generator import ReportGenerator
from qanta.reporting.plotting import plot_confusion


log = logging.get(__name__)


# Parameters were hand tuned
def create_gender_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=3)),
        ('lr', LogisticRegression(C=1000))
    ])


# Parameters were hand tuned
def create_category_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=3)),
        ('lr', LogisticRegression(C=1000))
    ])


def create_ans_type_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=3)),
        ('lr', LogisticRegression(C=1000))
    ])


def category_preprocess(data: pd.DataFrame):
    def strip_subcategory(category):
        if ':' in category:
            return category.split(':')[0]
        else:
            return category

    data['label'] = data['label'].map(strip_subcategory)
    return data

pipeline_creators = {
    'gender': create_gender_pipeline,
    'category': create_category_pipeline,
    'ans_type': create_ans_type_pipeline
}
preprocessors = {'category': category_preprocess}


def compute_features(all_questions, fold, class_type):
    features = []

    for page in all_questions:
        for q in all_questions[page]:
            if q.fold == fold:
                label = getattr(q, class_type)
                if not label or label == 'None':
                    continue
                for s, w, text in q.partials():
                    text = ' '.join(text)
                    features.append({
                        'sentence': s,
                        'word': w,
                        'text': text,
                        'label': label,
                        'qnum': q.qnum
                    })
    data = pd.DataFrame(features)
    if class_type in preprocessors:
        data = preprocessors[class_type](data)
    return data


def train_classifier(class_type, question_db=None):
    if question_db is None:
        question_db = QuestionDatabase(QB_QUESTION_DB)

    log.info("Training classifier: {}".format(class_type))
    all_questions = question_db.questions_with_pages()
    train = compute_features(all_questions, 'train', class_type)
    train_x = train['text']
    train_y = train['label']
    classifier = pipeline_creators[class_type]().fit(train_x, train_y)
    return classifier


def save_classifier(classifier, class_type):
    with safe_open(CLASSIFIER_PICKLE_PATH.format(class_type), 'wb') as f:
        pickle.dump(classifier, f)


def load_classifier(class_type):
    classifier_file = CLASSIFIER_PICKLE_PATH.format(class_type)
    with open(classifier_file, 'rb') as f:
        return pickle.load(f)


def compute_guess_properties():
    dataset = QuizBowlDataset(1)
    training_data = dataset.training_data()
    guess_ans_types = {}
    guess_categories = {}
    guess_genders = {}
    for page, aux_data in zip(training_data[1], training_data[2]):
        page = format_guess(page)

        ans_type = aux_data['ans_type']
        if ans_type != '' and ans_type != 'None':
            guess_ans_types[page] = ans_type

        category = aux_data['category']
        guess_categories[page] = category

        gender = aux_data['gender']
        if gender != '':
            guess_genders[page] = gender

    with open(CLASSIFIER_GUESS_PROPS, 'wb') as f:
        pickle.dump({
            'ans_type': guess_ans_types,
            'category': guess_categories,
            'gender': guess_genders
        }, f)


def load_guess_properties():
    with open(CLASSIFIER_GUESS_PROPS, 'rb') as f:
        return pickle.load(f)


def create_report(classifier, class_type, question_db=None):
    if question_db is None:
        question_db = QuestionDatabase(QB_QUESTION_DB)

    all_questions = question_db.questions_with_pages()
    train = compute_features(all_questions, 'train', class_type)
    train_x = train['text']
    train_y = train['label']
    dev = compute_features(all_questions, 'dev', class_type)
    dev_x = dev['text']
    dev_y = dev['label']
    train_score = classifier.score(train_x, train_y)
    dev_score = classifier.score(dev_x, dev_y)

    true_labels = dev['label'].values
    predicted_labels = classifier.predict(dev_x)

    cf_norm = '/tmp/norm_confusion.png'
    plot_confusion(
        'Row Normalized Confusion Matrix of {} Classification'.format(class_type),
        true_labels,
        predicted_labels,
        normalized=True
    )
    plt.savefig(cf_norm, format='png', dpi=200)
    plt.clf()
    plt.cla()
    plt.close()

    cf_unnorm = '/tmp/unnorm_confusion.png'
    plot_confusion(
        'Unnormalized Confusion Matrix of {} Classification'.format(class_type),
        true_labels,
        predicted_labels,
        normalized=False
    )
    plt.savefig(cf_unnorm, format='png', dpi=200)

    correct_by_position = '/tmp/correct_by_position.png'

    dev['prediction'] = pd.Series(predicted_labels)
    dev['correct'] = dev['prediction'] == dev['label']
    pd.pivot_table(
        dev, values=['text'], index=['sentence', 'correct'], aggfunc=lambda x: len(x)
    ).unstack(fill_value=0).plot.bar(
        title='Number of Questions Correct vs Sentence Number'
    )
    plt.xlabel('Sentence Number')
    plt.ylabel('Number Correct')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.gca().legend(handles, ['Number Incorrect', 'Number Correct'])
    plt.savefig(correct_by_position, format='png', dpi=200)

    report = ReportGenerator({
        'unnormalized_confusion_plot': cf_unnorm,
        'normalized_confusion_plot': cf_norm,
        'correct_by_position_plot': correct_by_position,
        'train_score': train_score,
        'dev_score': dev_score,
        'class_type': class_type
    }, 'classifier.md')
    output = safe_path(CLASSIFIER_REPORT_PATH.format(class_type))
    report.create(output)
    plt.clf()
    plt.cla()
    plt.close()
