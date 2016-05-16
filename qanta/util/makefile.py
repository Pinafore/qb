# pylint: disable=too-many-locals
import itertools
from jinja2 import Environment, FileSystemLoader
from qanta.util.constants import (
    GRANULARITIES, FEATURE_NAMES, FEATURES, MIN_APPEARANCES,
    NEGATIVE_WEIGHTS, COMPUTE_OPT_FEATURES, MEMORY_OPT_FEATURES, FOLDS)
from qanta.extractors.classifier import CLASSIFIER_FIELDS


QBDB = "data/questions.db"
FEATURE_LETTERS = ['g', 'i', 'l', 'm', 'd', 'a', 't', 'w']
FEATURE_LETTERS_WITHOUT_LM = ['g', 'i', 'm', 'd', 'a', 't', 'w']

# Path of wikifier input for expo files
WIKIFIER_EXPO_IN = "data/wikifier/data/expo_input"
WIKIFIER_EXPO_OUT = "data/wikifier/data/expo_output"


def base_feat(feat):
    if feat.startswith("ir"):
        return "ir"
    else:
        return feat


def feature_targets(granularity, features):
    return ' '.join("data/features/%s/%s.%s.parquet/_SUCCESS" % (fold, granularity, feature)
                    for (fold, feature) in itertools.product(FOLDS, features))


def vw_input_targets(folds, granularities, weights):
    return ' '.join(
        'data/vw_input/%s.%s.%i.vw_input' % (fold, granularity, weight)
        for fold, granularity, weight in itertools.product(folds, granularities, weights))


def get_feature_prereqs(feature):
    if feature == 'ir':
        req = ' '.join(
            'data/ir/whoosh_wiki_%i data/ir/whoosh_qb_%i' % (x, x) for x in [MIN_APPEARANCES])
    elif feature == 'wikilinks':
        req = 'data/wikifier/data/output'
    elif feature == 'lm':
        req = 'data/lm.txt'
    elif feature == 'classifier':
        req = ' '.join(['data/classifier/%s.pkl' % classifier for classifier in CLASSIFIER_FIELDS])
    elif feature == 'mentions':
        req = 'data/kenlm.binary'
    else:
        req = ''
    return req + ' data/guesses.db' if req != '' else 'data/guesses.db'


def generate():
    environment = Environment(loader=FileSystemLoader('qanta/makefile_templates'))
    environment.filters['feature_targets'] = feature_targets
    environment.filters['base_feat'] = base_feat
    environment.filters['get_feature_prereqs'] = get_feature_prereqs
    environment.filters['product'] = itertools.product
    environment.filters['vw_input_targets'] = vw_input_targets
    context = {
        'QBDB': QBDB,
        'MIN_APPEARANCES': MIN_APPEARANCES,
        'CLASSIFIER_FIELDS': CLASSIFIER_FIELDS,
        'GRANULARITIES': GRANULARITIES,
        'FOLDS': FOLDS,
        'FEATURES': FEATURES,
        'MEMORY_OPT_FEATURES': MEMORY_OPT_FEATURES,
        'COMPUTE_OPT_FEATURES': COMPUTE_OPT_FEATURES,
        'NEGATIVE_WEIGHTS': NEGATIVE_WEIGHTS,
        'FEATURE_NAMES': FEATURE_NAMES,
        'FEATURE_LETTERS': FEATURE_LETTERS,
        'FEATURE_LETTERS_WITHOUT_LM': FEATURE_LETTERS_WITHOUT_LM
    }
    o = open("Makefile", 'w')
    makefile_template = environment.get_template('makefile.template')
    o.write(makefile_template.render(context))
    o.close()
