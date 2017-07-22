from collections import Counter, defaultdict
from itertools import chain
import re
import pickle

import nltk

from qanta import logging
from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.wikipedia.cached_wikipedia import CachedWikipedia
from ingestion.title_finder import TitleFinder
from ingestion.page_assigner import PageAssigner

log = logging.get(__name__)

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

def add_question(connection, question_id, tournament, category, page,
                 content, answer, ans_type="", naqt=-1, protobowl="", fold="train"):
    c = connection.cursor()
    sentences = list(enumerate(sent_detector.tokenize(content)))
    c.executemany('INSERT INTO text VALUES (?, ?, ?)',
                  [(question_id, x, y) for x, y in
                   sentences])
    c.execute('INSERT INTO questions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
              (question_id, category, page, answer, tournament, ans_type, naqt, protobowl, fold))
    connection.commit()

    return sentences

kPRON = re.compile(r" \[[^\]]*\] ")

# Protobowl category mapping
kCATS = set(["Fine_Arts", "History", "Literature", "Other", "Science", "Social_Science"])
kCAT_MAP = dict((x, x) for x in kCATS)
kCAT_MAP['Mythology'] = "Social_Science:Mythology"
kCAT_MAP['Philosophy'] = "Social_Science:Philosophy"
kCAT_MAP['Religion'] = "Social_Science:Religion"
kCAT_MAP['Geography'] = "Social_Science:Geography"
kCAT_MAP['Trash'] = "Other:Trash"
kCAT_MAP['Current_Events'] = "Other:CE"

# Mapping NAQT categories into Protobowl's
kNAQT_MAP = {}
kNAQT_MAP["TH"] = "Social Studies"
kNAQT_MAP["SS"] = "Social Studies"
kNAQT_MAP["S:A"] = "Physics"
kNAQT_MAP["S:B"] = "Biology"
kNAQT_MAP["S:C"] = "Chemistry"
kNAQT_MAP["S:CS"] = "Mathematics"
kNAQT_MAP["S:ES"] = "Earth Science"
kNAQT_MAP["S:M"] = "Mathematics"
kNAQT_MAP["S:P"] = "Physics"
kNAQT_MAP["PH"] = "Social Studies"
kNAQT_MAP["L"] = "Literature"
kNAQT_MAP["H"] = "History"
kNAQT_MAP["FA:"] = "Fine Arts"

kYEAR = re.compile("[0-9]+")

kANSWER_PATTERN = ["\nanswer:", "\nAnswer:", "answer:", "Answer:", "asnwer:",
                   "answr:", "anwer:", "\nanswer"]


class NaqtQuestion:
    def __init__(self, raw):
        self.metadata = {}
        if raw.startswith("<QBML"):
            raw = raw.split(">", 1)[1]
        raw = raw.strip()
        header, rest = raw.split(">", 1)

        header = header.strip()
        assert header.startswith("<QUESTION "), header
        header = header.replace("<QUESTION ", "")

        # Extract metadata
        for ii in header.split('" '):
            field, val = ii.split('="')
            if field=="ID":
                self.metadata[field] = int(val)
            else:
                self.metadata[field] = val

        if "<PACKETSETS>" in rest:
            raw, rest = rest.split("<PACKETSETS>", 1)

        self.text = ""
        self.answer = ""
        # Not using a regexp because regexp doesn't have an rsplit
        # command and there is a clear precedence for how acceptable
        # answer patterns are
        for ii in kANSWER_PATTERN:
            if ii in raw:
                self.text, self.answer = raw.rsplit(ii, 1)
                self.answer = self.answer.strip().split('\n')[0]
                self.text = self.text.strip()
                break

        if "</PACKETSETS>" in rest:
            packets, topics = rest.split("</PACKETSETS>", 1)
        else:
            packets = ""
            topics = rest

        self.tournaments = "|".join(x for x in packets.split("\n")
                                    if x.strip())

        if "<TOPICS>" in rest:
            topics = topics.replace("</TOPICS>", "").strip()
            self.topics = {}
            for ii in topics.split("\n"):
                if ii.startswith("<TOPICS>") or ii.strip()=="":
                    continue
                first, rest = ii.split('ID="', 1)
                id, rest = rest.split('" TITLE="', 1)
                title, rest = rest.split('"', 1)
                self.topics[int(id)] = title

    def year(self):
        years = []

        for ii in [int(x) for x in kYEAR.findall(self.tournaments)]:
            if ii < 40:
                ii += 2000

            if ii > 70 and ii < 100:
                ii += 1900

            years.append(ii)



        if len(years) == 0:
            log.info("Bad year from %s" % self.tournaments)
            return 0
        elif "invitational series" in self.tournaments.lower():
            return years[0] / 10 + 1997

        elif len(years) > 2:
            years = years[:2]


        val = max(years)
        if val > 2016 and "invitational series" not in self.tournaments.lower():
            log.info("Crazy year %i" % val)
        return val

    @staticmethod
    def map_naqt(old_category):
        val = "Other"
        for ii in kNAQT_MAP:
            if old_category == ii or old_category.startswith("%s:" % ii):
                val = kNAQT_MAP[ii]
        if old_category.startswith("L:R:"):
            val = "Social Studies"
        if old_category == "":
            val = ""
        return val

    @staticmethod
    def naqt_reader(path):
        import os
        from glob import glob
        if os.path.isdir(path):
            files = glob("%s/*" % path)
        else:
            files = [path]

        for ii in files:
            with open(ii, encoding='utf-8') as infile:
                for jj in infile.read().split("</QUESTION>"):
                    if not 'KIND="TOSSUP"' in jj:
                        continue

                    if "<PACKETSET " in jj:
                        jj = jj.split("</PACKETSET>", 1)[1].strip()


                    q = NaqtQuestion(jj)

                    # Exclude computational questions
                    if "SUBJECT" in q.metadata and q.metadata["SUBJECT"].startswith("S:CO:"):
                        continue
                    yield q

    # TODO: handle NAQT special characters better
    @staticmethod
    def clean_naqt(text):
        if "<QUESTION" in text:
            text = text.split(">", 1)[1].strip()
        text = kPRON.sub(" ", text)
        text = text.replace("{", "").replace("}", "").replace(" (*) ", " ")
        text = text.replace("~", "")
        return text


def assign_fold(tournament, year):
    # Default assumption is that questions are (guesser) training data
    fold = "guesstrain"

    # Goal: 70% guess, 20% buzzer, 10% dev/test

    tourn = tournament.lower()

    # Years get messed up for invitational
    if "invitational series" in tourn:
        return "guesstrain"

    if "intramurals" in tourn or "winter" in tourn:
        fold = "guessdev"

    # ACF Fall, PACE, etc. are for training the buzzer
    if "acf" in tourn or "invitational" in tourn or "novice" in tourn:
        fold = "buzzertrain"

    if "nasat" in tourn or "bowl" in tourn or "open" in tourn:
        fold = "buzzertrain"

    if "pace" in tourn or "nsc" in tourn or "acf fall" in tourn:
        fold = "buzzerdev"

    # 2016 hsnct tournaments are dev
    if int(year) >= 2015:
        fold = "dev"

        if "high school championship" in tourn or "pace" in tourn or "nasat" in tourn:
            fold = "test"

    return fold


def map_protobowl(category, sub_cat):
    """
    Map protobowl categories to our categories
    """

    category = category.replace(" ", "_")
    if category in kCATS:
        if sub_cat:
            return "%s:%s" % (category, sub_cat.replace(" ", "_"))
        else:
            return category
    else:
        return kCAT_MAP[category]


def create_db(location):
    """
    Creates an empty QB database in this location, return database pointer
    """
    import sqlite3
    conn = sqlite3.connect(location)
    c = conn.cursor()

    # Create table
    c.execute('''CREATE TABLE questions
                   (id integer PRIMARY KEY, category text, page text,
                    answer text, tournament text, type text, naqt integer,
                    protobowl text, fold text)''')

    c.execute('''CREATE TABLE text
                   (question integer, sent integer, raw text,
                    foreign key(question) REFERENCES questions(id),
                    primary key(question, sent))''')

    # Save (commit) the changes
    conn.commit()

    return conn



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

    if flags.guess:
        log.info("Will guess page assignments")
    else:
        log.info("Will not guess page assignments")

    conn = create_db(flags.db)

    limit = None
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
    if flags.naqt_path:
        for qq in NaqtQuestion.naqt_reader(flags.naqt_path):
            if not qq.text:
                log.info("Bad question %s" % str(qq.metadata["ID"]))
                num_skipped += 1

            page = pa(qq.answer, tk(qq.text), naqt=qq.metadata["ID"])
            fold = assign_fold(qq.tournaments, qq.year())
            if page == "":
                norm = QuestionDatabase.normalize_answer(qq.answer)
                folds[fold] += 1

                if pa.is_ambiguous(norm):
                    ambiguous[norm][int(qq.metadata["ID"])] = qq.text
                else:
                    unmapped[norm] += 1


            add_question(conn, last_id, qq.tournaments,
                         NaqtQuestion.map_naqt(qq.metadata["SUBJECT"]),
                         page, qq.text, qq.answer, naqt=qq.metadata["ID"],
                         fold=fold)

            last_id += 1

            if last_id % 1000 == 0:
                log.info('{} {} {} {}'.format(last_id,
                                              qq.answer,
                                              page,
                                              qq.text))
                log.info(str(qq.tournaments))
                progress = pa.get_counts()
                for ii in progress:
                    log.info("MAP %s: %s" % (ii, progress[ii].most_common(5)))
                for ii in folds:
                    log.info("NAQT FOLD %s: %i" % (ii, folds[ii]))

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
