from jinja2 import Environment, FileSystemLoader
from util.constants import GRANULARITIES, FEATURES, FOLDS, MIN_APPEARANCES, NEGATIVE_WEIGHTS
from extractors.classifier import CLASSIFIER_FIELDS

VWOPT = {"full": "--early_terminate 100 -k -q gt -q ga -b 24 --loss_function logistic"}


QBDB = "data/questions.db"
FINAL_MOD = "full"

# Path of wikifier input for expo files
kWIKIFIER_EXPO_IN = "data/wikifier/data/expo_input"
kWIKIFIER_EXPO_OUT = "data/wikifier/data/expo_output"

assert FINAL_MOD in VWOPT, "Final model (%s) not in the set of VW models" % FINAL_MOD


def base_feat(feat):
    if feat.startswith("ir"):
        return "ir"
    else:
        return feat


def label_targets(granularity):
    return ' '.join("features/%s/%s.label.feat" % (fold, granularity) for fold in FOLDS)


def feature_targets(granularity, feature):
    return ' '.join("features/%s/%s.%s.feat" % (x, granularity, feature) for x in FOLDS)


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


if __name__ == "__main__":
    environment = Environment(loader=FileSystemLoader('makefile_templates'))
    environment.filters['label_targets'] = label_targets
    environment.filters['feature_targets'] = feature_targets
    environment.filters['base_feat'] = base_feat
    environment.filters['get_feature_prereqs'] = get_feature_prereqs
    context = {
        'QBDB': QBDB,
        'VWOPT': VWOPT,
        'MIN_APPEARANCES': MIN_APPEARANCES,
        'CLASSIFIER_FIELDS': CLASSIFIER_FIELDS,
        'GRANULARITIES': GRANULARITIES,
        'FOLDS': FOLDS,
        'FEATURES': FEATURES
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

    # Create label files with the desired weights
    for granularity in GRANULARITIES:
        for fold in FOLDS:
            o.write(" ".join("features/%s/%s.label." % (fold, granularity) + str(int(x))
                             for x in NEGATIVE_WEIGHTS))
            o.write(": features/%s/%s.label.feat\n" % (fold, granularity))
            o.write("\tpython util/reweight_labels.py $<")
            o.write("\n\n")

    # Generate the training data
    # (TODO): Perhaps create versions with different subsets of the features?
    # (TODO): Perhaps compress the training files after pasting them together?
    for granularity in GRANULARITIES:
        for fold in FOLDS:
            for weight in NEGATIVE_WEIGHTS:
                feature_filenames = ' '.join("features/%s/%s.%s.feat" %
                                             (fold, granularity, x) for x in FEATURES)
                o.write("features/%s/%s.%i.vw_input: " % (fold, granularity, int(weight)))
                o.write(feature_filenames)
                o.write(" features/%s/%s.label.%i\n\t" % (fold, granularity, int(weight)))
                o.write("paste features/%s/%s.label.%i " % (fold, granularity, int(weight)) +
                        " ".join("features/%s/%s.%s.feat"  % (fold, granularity, x)
                                 for x in FEATURES))
                if fold == "train":
                        temp_file = "vw_temp.%s.%s.%i" % (granularity, fold, weight)
                        o.write(" | gzip > %s\n" % temp_file)
                        o.write("\tpython scripts/shuffle.py %s $@\n" % temp_file)
                        o.write("\trm %s" % temp_file)
                else:
                        o.write(" | gzip > $@")
                o.write("\n\n")

    # Generate the VW model files and predictions
    for granularity in GRANULARITIES:
        for opt in VWOPT:
            for weight in NEGATIVE_WEIGHTS:
                # Model files
                model_file = "models/%s.%s.%i.vw" % (granularity, opt, int(weight))
                o.write("%s: " % model_file)
                assert "dev" in FOLDS, "Need training data to create models"
                o.write("features/%s/%s.%i.vw_input\n" %
                        ("dev", granularity, int(weight)))
                o.write("\tmkdir -p models\n")
                o.write("\tvw --compressed -d $< %s -f $@ " % VWOPT[opt])
                if "--ngram" in VWOPT[opt] or " -q " in VWOPT[opt] or " --quadratic" in VWOPT[opt]:
                    o.write("\n")
                else:
                    o.write("--invert_hash models/%s.%s.%i.read\n" %
                            (granularity, opt, int(weight)))
                    o.write("\tpython ")
                    o.write("util/sort_features.py models/%s.%s.%i.read" %
                            (granularity, opt, int(weight)))
                    o.write(" models/%s.%s.%i.sorted\n" % (granularity, opt, int(weight)))
                    o.write("\trm models/%s.%s.%i.read\n" % (granularity, opt, int(weight)))

                # Generate predictions
                for fold in FOLDS:
                    o.write("\nresults/%s/%s.%i.%s.pred: " % (fold, granularity, int(weight), opt))
                    input_file = "features/%s/%s.%i.vw_input" % (fold, granularity, int(weight))

                    o.write("%s %s\n" % (input_file, model_file))
                    o.write("\tmkdir -p results/%s\n" % fold)
                    o.write("\tvw --compressed -t -d %s -i %s " %
                            (input_file, model_file) + " -p $@\n")
                    o.write("\n")

                    o.write("results/%s/%s.%i.%s.buzz: " % (fold, granularity, int(weight), opt))
                    o.write(" results/%s/%s.%i.%s.pred " % (fold, granularity, int(weight), opt))
                    o.write("reporting/evaluate_predictions.py\n")
                    o.write("\tpython reporting/evaluate_predictions.py ")
                    o.write("--buzzes=$@ ")
                    o.write("--qbdb=%s " % QBDB)
                    o.write("--finals=results/%s/%s.%i.%s.final " %
                            (fold, granularity, int(weight), opt))
                    o.write("--question_out=results/%s/questions.csv " % fold)
                    o.write("--meta=features/%s/%s.meta " % (fold, granularity))
                    o.write("--perf=results/%s/%s.%i.%s.perf " %
                            (fold, granularity, int(weight), opt))
                    o.write("--neg_weight=%f " % weight)
                    o.write("--vw_config=%s " % opt)
                    o.write("--pred=$<")
                    o.write("\n\n")


    # Target for all predictions
    o.write("# Train all of the models")
    for granularity in GRANULARITIES:
        all_vw_models = []
        for opt in VWOPT:
            for weight in NEGATIVE_WEIGHTS:
                all_vw_models.append("models/%s.%s.%i.vw" % (granularity, opt, int(weight)))
        o.write("\n\nall_%s_models: " % granularity + " ".join(all_vw_models) + "\n\n")

    # Target for all buzzes
    o.write("# Buzz predictions for all models")
    for granularity in GRANULARITIES:
        all_buzzes = []
        for opt in VWOPT:
            for weight in NEGATIVE_WEIGHTS:
                for fold in FOLDS:
                    all_buzzes.append("results/%s/%s.%i.%s.buzz" %
                                      (fold, granularity, int(weight), opt))
        o.write("\n\nall_%s_buzz: " % granularity + " ".join(all_buzzes) + "\n\n")

    # Target for all performances
    o.write("# Get performance summaries\n")
    for fold in FOLDS:
        for granularity in GRANULARITIES:
            o.write("results/%s.%s.csv: " % (fold, granularity))
            all_perfs = []
            for opt in VWOPT:
                for weight in NEGATIVE_WEIGHTS:
                    all_perfs.append("results/%s/%s.%i.%s" % (fold, granularity, int(weight), opt))
            o.write(" ".join("%s.buzz" % x for x in all_perfs))
            o.write(" reporting/summarize.py\n\t")
            o.write("python reporting/summarize.py --output $@ -p ")
            o.write(" ".join("%s.perf" % x for x in all_perfs))
            o.write("\n\n")

    for fold in FOLDS:
        for granularity in GRANULARITIES:
            o.write("results/%s.%s.pdf: results/%s.%s.csv\n\t" %
                    (fold, granularity, fold, granularity))
            o.write("Rscript reporting/running_score.R $< $@\n\n")

    # plots of feature densities
    for granularity in GRANULARITIES:
        o.write("results/%s.features_cont.csv results/%s.features_disc.csv: " %
                (granularity, granularity))
        o.write("util/inspect_features.py ")
        o.write(" ".join("features/dev/%s.%s.feat" % (granularity, x)
                         for x in FEATURES))
        o.write("\n")
        o.write("\tpython util/inspect_features.py --feats ")
        o.write(" ".join("features/dev/%s.%s.feat" % (granularity, x)
                         for x in FEATURES))
        o.write(" --label features/dev/%s.label.feat" % granularity)
        o.write(" --output_cont results/%s.features_cont.csv" % granularity)
        o.write(" --output_disc results/%s.features_disc.csv" % granularity)
        o.write("\n\n")

        o.write("results/%s.features_disc.pdf results/%s.features_cont.pdf: " %
                (granularity, granularity))
        o.write("results/%s.features_cont.csv results/%s.features_disc.csv " %
                (granularity, granularity))
        o.write("util/density_plots.R\n")
        o.write("\tRscript util/density_plots.R %s\n\n" % granularity)

    # Expo wikifier
    o.write("%s: data/expo.csv util/wikification.py\n" % kWIKIFIER_EXPO_IN)
    o.write("\trm -rf $@\n")
    o.write("\tmkdir -p $@\n")
    o.write("\tpython util/wikification.py --output_directory=$@")
    o.write(" --database='' --min_pages=-1 --expo=data/expo.csv\n\n")

    o.write("%s: %s\n" % (kWIKIFIER_EXPO_OUT, kWIKIFIER_EXPO_IN))
    o.write("\trm -rf $@\n")
    o.write("\tmkdir -p $@\n")
    o.write("\tcp lib/STAND_ALONE_NO_INFERENCE.xml ")
    o.write("data/wikifier/STAND_ALONE_NO_INFERENCE.xml\n")
    o.write("\t(cd data/wikifier && java -Xmx10G -jar ")
    o.write("wikifier-3.0-jar-with-dependencies.jar ")
    o.write("-annotateData %s %s " %
            (kWIKIFIER_EXPO_IN.replace("data/wikifier/", ""),
             kWIKIFIER_EXPO_OUT.replace("data/wikifier/", "")))
    o.write("false STAND_ALONE_NO_INFERENCE.xml)\n")
    o.write("\tcp $@/* data/wikifier/data/output\n\n")

    # Expo features
    o.write("features/expo/word.label.feat: ")
    o.write("extract_expo_features.py ")
    o.write(" ".join(sorted(feature_prereq)))
    o.write(" %s" % kWIKIFIER_EXPO_OUT)
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
        model_file = "models/sentence.%s.%i.vw" % (FINAL_MOD, weight)
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
        o.write("--perf=results/expo/word.%i.%s.perf " % (int(weight), FINAL_MOD))
        o.write("--neg_weight=%f " % weight)
        o.write("--vw_config=%s " % FINAL_MOD)
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
