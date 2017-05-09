from collections import Counter, defaultdict
from itertools import chain

import nltk

from qanta import logging
from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.wikipedia.cached_wikipedia import CachedWikipedia
from ingestion.title_finder import TitleFinder

log = logging.get(__name__)

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

def add_question(connection, question_id, tournament, category, page,
                 content, answer, ans_type="", naqt=-1, protobowl="", fold="train"):
    c = connection.cursor()

    c.executemany('INSERT INTO text VALUES (?, ?, ?)',
                  [(question_id, x, y) for x, y in
                   enumerate(sent_detector.tokenize(content))])
    c.execute('INSERT INTO questions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
              (question_id, category, page, answer, tournament, ans_type, naqt, protobowl, fold))
    connection.commit()

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
            self.metadata[field] = val

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

        packets, topics = rest.split("</PACKETSETS>", 1)

        self.tournaments = "|".join(x for x in packets.split("\n")
                                    if x.strip())

        topics = topics.replace("</TOPICS>", "").strip()
        self.topics = {}
        for ii in topics.split("\n"):
            if ii.startswith("<TOPICS>") or ii.strip()=="":
                continue
            first, rest = ii.split('ID="', 1)
            id, rest = rest.split('" TITLE="', 1)
            title, rest = rest.split('"', 1)
            self.topics[int(id)] = title

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
        if os.path.isdir(path):
            files = glob("%s/*" % path)
        else:
            files = [path]

        for ii in files:
            with open(ii, encoding='utf-8') as infile:
                for jj in infile.read().split("</QUESTION>"):
                    if not 'KIND="TOSSUP"' in jj:
                        continue


                    q = NaqtQuestion(jj)

                    # Exclude computational questions
                    if q.metadata["SUBJECT"].startswith("S:CO:"):
                        continue
                    yield q

def assign_fold(tournament, year):
    # Default assumption is that questions are (guesser) training data
    fold = "train"

    tourn = tournament.lower()
    # ACF Fall, PACE, etc. are for training the buzzer
    if "acf" in tourn and ("fall" in tourn or "novice" in tourn):
        fold = "buzzer"

    if "pace" in tourn and "nsc" in tourn:
        fold = "buzzer"

    if "nasat" in tourn:
        fold = "buzzer"

    # 2016 hsnct tournaments are dev
    if int(year) > 2015 or "high school championship" in tourn:
        fold = "dev"

    # HSNCT 2016 is test
    if int(year) == 2016 and "high school championship" in tourn:
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

class PageAssigner:
    def __init__(self, normalize_func=lambda x: x):
        from collections import defaultdict

        self._unambiguous = {}
        self._ambiguous = defaultdict(dict)
        self._direct = defaultdict(dict)

        self._normalize = normalize_func

    def load_unambiguous(self, filename):
        with open(filename) as infile:
            for ii in infile:
                if not ii.strip():
                    continue
                try:
                    answer, page = ii.strip().split('\t')
                except ValueError:
                    log.info("Bad unambiguous line in %s: %s" % (filename, ii))
                self._unambiguous[answer] = page

    def load_ambiguous(self, filename):
        with open(filename) as infile:
            for ii in infile:
                if not ii.strip():
                    continue
                fields = ii.strip().split('\t')

                if not (len(fields) == 2 or len(fields) == 3):
                    log.info("Bad ambiguous line in %s: %s" % (filename, ii))
                    continue
                if len(fields) == 3:
                    answer, page, words = fields
                else:
                    answer, page = fields
                    words = ""

                words = words.split(":")
                self._ambiguous[answer][page] = words

    def load_direct(self, filename):
        with open(filename) as infile:
            for ii in infile:
                if not ii.strip():
                    continue
                try:
                    ext_id, page, answer = ii.strip().split('\t')
                except ValueError:
                    log.info("Bad direct line in %s: %s" % (filename, ii))
                self._direct[ext_id] = page

    def __call__(self, answer, text, pb="", naqt=-1):
        normalize = self._normalize(answer)

        if pb in self._direct:
            return self._direct[pb]

        if naqt in self._direct:
            return self._direct[naqt]

        if normalize in self._unambiguous:
            return self._unambiguous[normalize]

        if normalize in self._ambiguous:
            default = [x for x in self._ambiguous[normalize] if
                       len(self._ambiguous[normalize]) == 0]
            assert len(default) <= 1, "%s has more than one default" % normalize
            assert len(default) < len(self._ambiguous[normalize]), "%s only has default" % normalize

            # See if any words match
            words = None
            for jj in self._ambiguous[normalize]:
                for ww in self._ambiguous[normalize][jj]:
                    if words is None:
                        words = set(text)
                    if ww in [x.lower() for x in words]:
                        return jj

            # Return default if there is one
            if len(default) == 1:
                return default[0]
            else:
                return ''

        # Give up if we can't find answer
        return ''


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

    flags = parser.parse_args()
    tf = TitleFinder(flags.wiki_title, CachedWikipedia(),
                     QuestionDatabase.normalize_answer)

    # Load page assignment information
    pa = PageAssigner(QuestionDatabase.normalize_answer)
    for ii in glob("%s/*" % flags.ambiguous_path):
        pa.load_ambiguous(ii)
    for ii in glob("%s/*" % flags.unambiguous_path):
        pa.load_unambiguous(ii)
    for ii in glob("%s/*" % flags.direct_path):
        pa.load_direct(ii)

    conn = create_db(flags.db)

    unmapped = Counter()
    last_id = 0
    num_skipped = 0
    if flags.naqt_path:
        for qq in NaqtQuestion.naqt_reader(flags.naqt_path):
            if not qq.text:
                log.info("Bad question %s" % str(qq.metadata["ID"]))
                num_skipped += 1

            page = pa(qq.answer, tk(qq.text), naqt=qq.metadata["ID"])
            if page == "":
                unmapped[QuestionDatabase.normalize_answer(qq.answer)] += 1
            add_question(conn, last_id, qq.tournaments,
                         NaqtQuestion.map_naqt(qq.metadata["SUBJECT"]),
                         page, qq.text, qq.answer, naqt=qq.metadata["ID"],
                         fold=assign_fold(qq.tournament, qq.year))
            last_id += 1

            if last_id % 1000 == 0:
                log.info('{} {} {} {}'.format(last_id,
                                              qq.answer,
                                              page,
                                              qq.text))
                log.info(str(qq.tournaments))

    if flags.protobowl:
        with open(flags.protobowl) as infile:
            import json
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
                add_question(conn, last_id, question["tournament"], category,
                             page, question["question"], ans, protobowl=pid,
                             fold=assign_fold(question["tournament"],
                                              question["year"]))
                if page == "":
                    norm = QuestionDatabase.normalize_answer(ans)
                    unmapped[norm] += 1
                last_id += 1
    log.info("Added %i, skipped %i" % (last_id, num_skipped))

    if not os.path.exists(flags.wiki_title):
        urllib.request.urlretrieve("http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-all-titles-in-ns0.gz",
                                   flags.wiki_title)

    guesses = tf.best_guess(unmapped)

    wiki_total = Counter()
    wiki_answers = defaultdict(set)
    for ii in guesses:
        page = guesses[ii]
        wiki_total[page] += unmapped[ii]
        wiki_answers[page].add(ii)

    for ii in [x for x in unmapped if not x in guesses]:
        wiki_answers[''].add(ii)

    # Todo: sort by (redirected) wikipedia page frequency
    with open(flags.unmapped_report, 'w') as outfile:
        for pp, cc in wiki_total.most_common():
            for kk in wiki_answers[pp]:
                # TODO: sort by frequency
                outfile.write("%s\t%s\n" % (kk, pp))
