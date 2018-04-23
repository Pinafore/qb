from collections import Counter, defaultdict
import pickle

from qanta import qlogging
from qanta.datasets.quiz_bowl import QuestionDatabase
from ingestion.title_finder import TitleFinder
from ingestion.page_assigner import PageAssigner

log = qlogging.get(__name__)



if __name__ == "__main__":
    import argparse
    import datetime
    import os
    from glob import glob
    from nltk.tokenize.treebank import TreebankWordTokenizer
    tk = TreebankWordTokenizer().tokenize

    parser = argparse.ArgumentParser(description='Import questions')
    parser.add_argument('--naqt_path', type=str, default='data/questions/naqt/2017')
    parser.add_argument('--protobowl', type=str,
                        default='data/questions/protobowl/questions-05-05-2017.json')
    parser.add_argument('--unmapped_report', type=str, default="unmapped.txt")
    parser.add_argument('--ambig_report', type=str, default="ambiguous.txt")
    parser.add_argument('--limit_set', type=str,
                        default="data/external/wikipedia-titles.pickle")
    parser.add_argument('--direct_path', type=str,
                        default='data/internal/page_assignment/direct/')
    parser.add_argument('--ambiguous_path', type=str,
                        default='data/internal/page_assignment/ambiguous/')
    parser.add_argument('--unambiguous_path', type=str,
                        default='data/internal/page_assignment/unambiguous/')
    parser.add_argument('--db', type=str, default='data/internal/%s.db' %
                        datetime.date.today().strftime("%Y_%m_%d"))
    parser.add_argument('--wiki_title',
                        default="data/enwiki-latest-all-titles-in-ns0.gz",
                        type=str)
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--guess', dest='guess', action='store_true')
    feature_parser.add_argument('--no-guess', dest='guess', action='store_false')
    parser.set_defaults(guess=True)
    parser.add_argument('--csv_out', default="protobowl.csv", type=str)

    flags = parser.parse_args()

    try:
        limit = pickle.load(open(flags.limit_set, 'rb'))
    except IOError:
        log.info("Failed to load limit set from %s" % flags.limit_set)
        limit = None

    # Load page assignment information
    pa = PageAssigner(QuestionDatabase.normalize_answer,
                      limit)
    for ii in glob("%s/*" % flags.ambiguous_path):
        pa.load_ambiguous(ii)
    for ii in glob("%s/*" % flags.unambiguous_path):
        pa.load_unambiguous(ii)
    for ii in glob("%s/*" % flags.direct_path):
        pa.load_direct(ii)

    ambiguous = defaultdict(dict)
    unmapped = Counter()
    folds = Counter()
    last_id = 0
    num_skipped = 0

    if flags.protobowl:
        with open(flags.protobowl) as infile, open(flags.csv_out, 'w') as outfile:
            import json
            from csv import DictWriter
            o = DictWriter(outfile, ["id", "sent", "text", "ans", "page", "fold"])
            o.writeheader()
            for ii in infile:
                try:
                    question = json.loads(ii)
                except ValueError:
                    log.info("Parse error: %s" % ii)
                    num_skipped += 1
                    continue

                pid = question["_id"]["$oid"]
                ans = question["answer"]
                category = map_protobowl(question['category'],
                                         question.get('subcategory', ''))
                page = pa(ans, tk(question["question"]), pb=pid)
                fold = assign_fold(question["tournament"],
                                   question["year"])
                sents = add_question(conn, last_id, question["tournament"], category,
                             page, question["question"], ans, protobowl=pid,
                             fold=fold)

                for ii, ss in sents:
                    o.writerow({"id": pid,
                                 "sent": ii,
                                 "text": ss,
                                 "ans": ans,
                                 "page": page,
                                 "fold": fold})

                if page == "":
                    norm = QuestionDatabase.normalize_answer(ans)
                    if pa.is_ambiguous(norm):
                        ambiguous[norm][pid] = question["question"]
                    else:
                        unmapped[norm] += 1
                else:
                    folds[fold] += 1
                last_id += 1

                if last_id % 1000 == 0:
                    progress = pa.get_counts()
                    for ii in progress:
                        log.info("MAP %s: %s" % (ii, progress[ii].most_common(5)))
                    for ii in folds:
                        log.info("PB FOLD %s: %i" % (ii, folds[ii]))

    log.info("Added %i, skipped %i" % (last_id, num_skipped))

    if flags.guess:
        if not os.path.exists(flags.wiki_title):
            import urllib
            urllib.request.urlretrieve("http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-all-titles-in-ns0.gz",
                                    flags.wiki_title)

        tf = TitleFinder(flags.wiki_title, CachedWikipedia(),
                        pa.known_pages(),
                        QuestionDatabase.normalize_answer)

        guesses = tf.best_guess(unmapped)
    else:
        guesses = dict((x, "") for x in unmapped)

    wiki_total = Counter()
    wiki_answers = defaultdict(set)
    for ii in guesses:
        page = guesses[ii]
        wiki_total[page] += unmapped[ii]
        wiki_answers[page].add(ii)

    for ii in [x for x in unmapped if not x in guesses]:
        wiki_answers[''].add(ii)

    with open(flags.unmapped_report, 'w') as outfile:
        for pp, cc in wiki_total.most_common():
            for kk in wiki_answers[pp]:
                if not pa.is_ambiguous(kk):
                    # TODO: sort by frequency
                    outfile.write("%s\t%s\t%i\t%i\n" %
                                  (kk, pp, cc, unmapped[kk]))

    with open(flags.ambig_report, 'w') as outfile:
        for aa in sorted(ambiguous, key=lambda x: len(ambiguous[x]), reverse=True):
            for ii in ambiguous[aa]:
                outfile.write("%s\t%s\t%s\n" % (str(ii), aa, ambiguous[aa][ii]))
