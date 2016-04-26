from glob import glob
from qanta.util.qdb import QuestionDatabase

import codecs
import os

import nltk

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

kNAQT_START = 212895

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

# Convienence functions for reading NAQT data

def map_naqt(old_category, text, wiki_page):
    val = "Other"
    for ii in kNAQT_MAP:
        if old_category == ii or old_category.startswith("%s:" % ii):
            val = kNAQT_MAP[ii]
    if old_category.startswith("L:R:"):
        val = "Social Studies"
    if old_category == "":
        val = ""
    return val

def add_question(connection, question_id, tournament, category, page,
                 content, answer, ans_type="", naqt=-1, fold="dev"):
    c = connection.cursor()

    c.executemany('INSERT INTO text VALUES (?, ?, ?)',
                  [(question_id, x, y) for x, y in
                   enumerate(sent_detector.tokenize(content.replace('?', '')))])
    c.execute('INSERT INTO questions VALUES (?, ?, ?, ?, ?, ?, ?, ?, "")',
              (question_id, category, page, answer, tournament, ans_type, naqt, fold))
    connection.commit()

def naqt_reader(path):
    if os.path.isdir(path):
        files = glob("%s/*" % path)
    else:
        files = [path]

    for ii in files:
        for jj in codecs.open(ii, encoding='utf-8').read().split("</QUESTION>"):
            if not 'KIND="TOSSUP"' in jj:
                continue
            q = NaqtQuestion(jj)

            # Exclude computational questions
            if q.metadata["SUBJECT"].startswith("S:CO:"):
                continue
            yield q


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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Import questions')
    parser.add_argument('--naqt_path', type=str)
    parser.add_argument('--db', type=str, default='data/questions.db')

    flags = parser.parse_args()

    qdb = QuestionDatabase(flags.db)
    conn = qdb._conn
    answer_map = qdb.answer_map()

    # Find existing naqt questions
    c = conn.cursor()
    command = 'SELECT naqt FROM questions WHERE naqt >= 0;'
    c.execute(command)
    existing = set(int(x[0]) for x in c)

    num_skipped = 0
    last_id = kNAQT_START
    if flags.naqt_path:
        for qq in naqt_reader(flags.naqt_path):
            if qq.answer in answer_map and len(answer_map[qq.answer]) == 1:
                page = answer_map[qq.answer].keys()[0]
            else:
                page = ""

            if not qq.text:
                print("Bad question %s" % str(qq.metadata["ID"]))

            if int(qq.metadata["ID"]) in existing:
                num_skipped += 1
                continue
            else:
                last_id += 1

            add_question(conn, last_id, qq.tournaments,
                         map_naqt(qq.metadata["SUBJECT"], qq.text, page),
                         page, qq.text, qq.answer, naqt=qq.metadata["ID"])

            if last_id % 1000 == 0:
                print(answer_map[qq.answer])
                print(last_id, qq.answer, page, qq.text)
                print(qq.tournaments)

    print("Added %i, skipped %i" % (last_id - kNAQT_START, num_skipped))
    qdb.prune_text()
