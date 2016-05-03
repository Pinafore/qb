# pylint: disable=too-many-locals
import itertools
from jinja2 import Environment, FileSystemLoader
from qanta.util.constants import (
    GRANULARITIES, FEATURE_NAMES, FEATURES, MIN_APPEARANCES,
    NEGATIVE_WEIGHTS, COMPUTE_OPT_FEATURES, MEMORY_OPT_FEATURES, FOLDS)
from extractors.classifier import CLASSIFIER_FIELDS


QBDB = "data/questions.db"
FEATURE_LETTERS = ['g', 'i', 'l', 'm', 'd', 'a', 't', 'w']

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


def main():
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
        'FEATURE_LETTERS': FEATURE_LETTERS
    }
    o = open("Makefile", 'w')
    feature_prereq = set()

    makefile_template = environment.get_template('makefile.template')
    feature_prereq.add("data/deep/params")
    o.write(makefile_template.render(context))

    # Generate rules for generating the features
    for granularity in GRANULARITIES:
        for feature in FEATURES:

            if feature == "ir":
                feature_prereq.add("data/ir/whoosh_wiki_%i" % MIN_APPEARANCES)
                feature_prereq.add("data/ir/whoosh_qb_%i" % MIN_APPEARANCES)

            if feature == "wikilinks":
                feature_prereq.add("data/wikifier/data/output")

            if feature == "lm":
                feature_prereq.add("data/lm.txt")

            if feature == "classifier":
                for classifier in CLASSIFIER_FIELDS:
                    fname = "data/classifier/%s.pkl" % classifier
                    feature_prereq.add(fname)

            if feature == "mentions":
                feature_prereq.add("data/kenlm.binary")

    # Target for all predictions
    o.write("# Train all of the models")
    for granularity in GRANULARITIES:
        all_vw_models = []
        for weight in NEGATIVE_WEIGHTS:
            all_vw_models.append("data/models/%s.full.%i.vw" % (granularity, int(weight)))
        o.write("\n\nall_%s_models: " % granularity + " ".join(all_vw_models) + "\n\n")

    # Target for all buzzes
    o.write("# Buzz predictions for all models")
    for granularity in GRANULARITIES:
        all_buzzes = []
        for weight in NEGATIVE_WEIGHTS:
            for fold in FOLDS:
                all_buzzes.append("data/results/%s/%s.%i.summary.json" %
                                  (fold, granularity, int(weight)))
        o.write("\n\nall_%s_buzz: " % granularity + " ".join(all_buzzes) + "\n\n")

    # Target for all performances
    o.write("# Get performance summaries\n")
    for fold in FOLDS:
        for granularity in GRANULARITIES:
            o.write("data/results/%s.%s.csv: " % (fold, granularity))
            all_perfs = []
            for weight in NEGATIVE_WEIGHTS:
                all_perfs.append(
                    "data/results/%s/%s.%i.full" % (fold, granularity, int(weight)))
            o.write(" ".join("%s.buzz" % x for x in all_perfs))
            o.write(" reporting/summarize.py\n\t")
            o.write("python3 reporting/summarize.py --output $@ -p ")
            o.write(" ".join("%s.perf" % x for x in all_perfs))
            o.write("\n\n")

    for fold in FOLDS:
        for granularity in GRANULARITIES:
            o.write("data/results/%s.%s.pdf: data/results/%s.%s.csv\n\t" %
                    (fold, granularity, fold, granularity))
            o.write("Rscript reporting/running_score.R $< $@\n\n")

    # plots of feature densities
    for granularity in GRANULARITIES:
        o.write("data/results/%s.features_cont.csv data/results/%s.features_disc.csv: " %
                (granularity, granularity))
        o.write("util/inspect_features.py ")
        o.write(" ".join("data/features/dev/%s.%s.feat" % (granularity, x)
                         for x in FEATURES))
        o.write("\n")
        o.write("\tpython3 util/inspect_features.py --feats ")
        o.write(" ".join("data/features/dev/%s.%s.feat" % (granularity, x)
                         for x in FEATURES))
        o.write(" --label data/features/dev/%s.label.feat" % granularity)
        o.write(" --output_cont data/results/%s.features_cont.csv" % granularity)
        o.write(" --output_disc data/results/%s.features_disc.csv" % granularity)
        o.write("\n\n")

        o.write("data/results/%s.features_disc.pdf results/%s.features_cont.pdf: " %
                (granularity, granularity))
        o.write("data/results/%s.features_cont.csv results/%s.features_disc.csv " %
                (granularity, granularity))
        o.write("util/density_plots.R\n")
        o.write("\tRscript util/density_plots.R %s\n\n" % granularity)

    # Expo wikifier
    o.write("%s: data/expo.csv util/wikification.py\n" % WIKIFIER_EXPO_IN)
    o.write("\trm -rf $@\n")
    o.write("\tmkdir -p $@\n")
    o.write("\tpython util/wikification.py --output_directory=$@")
    o.write(" --database='' --min_pages=-1 --expo=data/expo.csv\n\n")

    o.write("%s: %s\n" % (WIKIFIER_EXPO_OUT, WIKIFIER_EXPO_IN))
    o.write("\trm -rf $@\n")
    o.write("\tmkdir -p $@\n")
    o.write("\tcp lib/STAND_ALONE_NO_INFERENCE.xml ")
    o.write("data/wikifier/STAND_ALONE_NO_INFERENCE.xml\n")
    o.write("\t(cd data/wikifier && java -Xmx10G -jar ")
    o.write("wikifier-3.0-jar-with-dependencies.jar ")
    o.write("-annotateData %s %s " %
            (WIKIFIER_EXPO_IN.replace("data/wikifier/", ""),
             WIKIFIER_EXPO_OUT.replace("data/wikifier/", "")))
    o.write("false ../../lib/STAND_ALONE_GUROBI.xml)\n")
    o.write("\tcp $@/* data/wikifier/data/output\n\n")

    # Expo features
    o.write("features/expo/word.label.feat: ")
    o.write("extract_expo_features.py ")
    o.write(" ".join(sorted(feature_prereq)))
    o.write(" %s" % WIKIFIER_EXPO_OUT)
    o.write("\n\tmkdir -p features/expo")
    o.write("\n\tmkdir -p results/expo")
    o.write("\n\trm -f data/expo_guess.db")
    o.write("\n\tpython extract_expo_features.py")
    o.write("\n\n")

    # Expo labels
    o.write(" ".join("features/expo/word.label.%i" % x for x in NEGATIVE_WEIGHTS))
    o.write(": features/expo/word.label.feat\n")
    o.write("\tpython util/reweight_labels.py $<\n\n")

    for weight in NEGATIVE_WEIGHTS:
        o.write("features/expo/expo.%i.vw_input: features/expo/word.label.%i"
                % (weight, weight))
        o.write("\n\t")
        o.write("paste features/expo/word.label.%i" % weight)
        for feature in FEATURES:
            o.write(" features/expo/word.%s.feat" % feature)
        o.write("| gzip > $@\n\n")

    # produce predictions and buzzes
    for weight in NEGATIVE_WEIGHTS:
        # predictions
        input_file = "features/expo/expo.%i.vw_input" % weight
        model_file = "models/sentence.full.%i.vw" % weight
        o.write("results/expo/expo.%i.pred: %s" % (weight, input_file))
        o.write(" %s" % model_file)
        o.write("\n")
        o.write("\tvw --compressed -t -d %s -i %s " %
                (input_file, model_file) + " -p $@ ")
        o.write("--audit > results/expo/expo.%i.audit\n\n" % ww)

        # Buzzes
        o.write("results/expo/expo.%i.buzz: results/expo/expo.%i.pred\n" %
                (weight, weight))
        o.write("\tmkdir -p results/expo\n")
        o.write("\tpython reporting/evaluate_predictions.py ")
        o.write("--buzzes=$@ ")
        o.write("--qbdb=%s " % QBDB)
        o.write("--question_out='' ")
        o.write("--meta=features/expo/word.meta ")
        o.write("--perf=results/expo/word.%i.full.perf " % (int(weight)))
        o.write("--neg_weight=%f " % weight)
        o.write("--vw_config=full ")
        o.write("--expo=data/expo.csv ")
        o.write("--finals=results/expo/expo.%i.final " % weight)
        o.write("--pred=$<")
        o.write("\n\n")

    # target for running the demo
    for weight in NEGATIVE_WEIGHTS:
        o.write("demo%i: results/expo/expo.%i.buzz\n" % (weight, weight))
        o.write("\tpython util/buzzer.py")
        o.write(" --questions=results/expo/questions.csv")
        o.write(" --buzzes=results/expo/expo.%i.buzz" % weight)
        o.write(" --output=results/expo/competition.csv")
        o.write(" --finals=results/expo/expo.%i.final" % weight)
        o.write(" --power=data/expo_power.csv")
        o.write('\n\n')

    o.write('\n\n')
    o.write("clean:\n")
    o.write("\trm -rf data/guesses.db features data/deep/params\n")
    o.write("\trm -rf data/classifier/*.pkl data/wikifier/data/input data/wikifier/data/output")


if __name__ == "__main__":
    main()
