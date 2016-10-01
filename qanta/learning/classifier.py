import pickle

import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest

from xgboost import XGBClassifier

from qanta import logging
from qanta.util.io import safe_open, safe_path
from qanta.util.qdb import QuestionDatabase
from qanta.util.constants import CLASSIFIER_PICKLE_PATH, CLASSIFIER_REPORT_PATH
from qanta.util.sklearn import DFColumnTransformer, CSCTransformer
from qanta.util.environment import QB_QUESTION_DB
from qanta.reporting.report_generator import ReportGenerator
from qanta.reporting.plotting import plot_confusion


log = logging.get(__name__)


# Parameters were hand tuned
def create_gender_pipeline():
    return Pipeline([
        ('df_select', DFColumnTransformer(['text'])),
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=3)),
        ('select_k', SelectKBest(k=750)),
        ('csc', CSCTransformer()),
        ('xgb', XGBClassifier(max_depth=8))
    ])


# Parameters were hand tuned
def create_category_pipeline():
    return Pipeline([
        ('df_select', DFColumnTransformer(['text'])),
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=3)),
        ('select_k', SelectKBest(k=1300)),
        ('csc', CSCTransformer()),
        ('xgb', XGBClassifier(max_depth=12, n_estimators=400))
    ])


def create_ans_type_pipeline():
    return Pipeline([
        ('df_select', DFColumnTransformer(['text'])),
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=3)),
        ('select_k', SelectKBest(k=1250)),
        ('csc', CSCTransformer()),
        ('xgb', XGBClassifier(max_depth=13))
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
    classifier = pipeline_creators[class_type]().fit(train, train['label'])
    return classifier


def save_classifier(classifier, class_type):
    with safe_open(CLASSIFIER_PICKLE_PATH.format(class_type), 'wb') as f:
        pickle.dump(classifier, f)


def load_classifier(class_type):
    classifier_file = CLASSIFIER_PICKLE_PATH.format(class_type)
    with open(classifier_file, 'rb') as f:
        return pickle.load(f)


def create_report(classifier, class_type, question_db=None):
    if question_db is None:
        question_db = QuestionDatabase(QB_QUESTION_DB)

    all_questions = question_db.questions_with_pages()
    train = compute_features(all_questions, 'train', class_type)
    dev = compute_features(all_questions, 'dev', class_type)
    train_score = classifier.score(train, train['label'])
    dev_score = classifier.score(dev, dev['label'])

    true_labels = dev['label'].values
    predicted_labels = classifier.predict(dev)

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
