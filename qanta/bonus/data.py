import re
import os
import regex
import pickle
import spacy
import json
from tqdm import tqdm

from multiprocessing import Pool
from multiprocessing.util import Finalize
from functools import partial

from torchtext import data

from qanta.util.environment import BONUS_ANSWER_PAGES, BONUS_PAIRS_JSON
from qanta.datasets.quiz_bowl import BonusQuestionDatabase

GROUP_LENGTH = 0


def _split_doc(doc):
    """Given a doc, split it into chunks (by paragraph)."""
    curr = []
    curr_len = 0
    for split in regex.split(r"\n+", doc):
        split = split.strip()
        if len(split) == 0:
            continue
        # Maybe group paragraphs together until we hit a length limit
        if len(curr) > 0 and curr_len + len(split) > GROUP_LENGTH:
            return " ".join(curr)
            curr = []
            curr_len = 0
        curr.append(split)
        curr_len += len(split)
    if len(curr) > 0:
        return " ".join(curr)


class WikiPage:
    def __init__(self, title, content, links, summary, categories, url, pageid):
        self.title = title
        self.content = content
        self.links = links
        self.summary = summary
        self.categories = categories
        self.url = url
        self.pageid = pageid


class BonusPair:
    def __init__(
        self, qnum, document_num, query_num, document, query, answer, start, end
    ):
        self.qnum = qnum
        self.document_num = document_num
        self.query_num = query_num
        self.document = document
        self.query = query
        self.answer = answer
        self.start = start
        self.end = end


class BonusPairsDataset:
    """Each entry contains the following:
        - qid & question number (0, 1, 2)
        - wikipage content
        - question or query
        - start & end position of the answer
        - answer
    """

    def __init__(self, save_dir=BONUS_PAIRS_JSON):
        self.paragraph_length = 50
        if os.path.isfile(save_dir):
            with open(save_dir, "r") as f:
                self.examples = json.load(f)
        else:
            self.examples = self.create(save_dir)

    def create(self, save_dir):
        if os.path.isfile(save_dir):
            with open(save_dir, "r") as f:
                self.examples = json.load(_examples, f)
            return

        # load bonus questions
        bonus_questions = BonusQuestionDatabase().all_questions()

        # load wikipages
        with open(BONUS_ANSWER_PAGES, "rb") as f:
            wikipages = pickle.load(f)
        wikipages = {k: WikiPage(*v) for k, v in wikipages.items() if v is not None}

        # qid, wikipage number (0, 1, 2), question number (0, 1, 2),
        # wikipage content, query, answer as wikipage title
        examples = []
        for q in bonus_questions.values():
            qnum = q.qnum
            wikipage_0 = wikipages.get(q.pages[0], None)
            wikipage_1 = wikipages.get(q.pages[1], None)
            wikipage_2 = wikipages.get(q.pages[2], None)
            if wikipage_1 is not None:
                if wikipage_0 is not None and wikipage_1.title in wikipage_0.links:
                    examples.append(
                        [qnum, 0, 1, wikipage_0.content, q.texts[1], wikipage_1.title]
                    )
            if wikipage_2 is not None:
                if wikipage_0 is not None and wikipage_2.title in wikipage_0.links:
                    examples.append(
                        [qnum, 0, 2, wikipage_0.content, q.texts[2], wikipage_2.title]
                    )
                if wikipage_1 is not None and wikipage_2.title in wikipage_1.links:
                    examples.append(
                        [qnum, 1, 2, wikipage_1.content, q.texts[2], wikipage_2.title]
                    )

        """ tokenize pages and queries """
        nlp = spacy.load("en")
        _, _, _, pages, queries, _ = list(zip(*examples))
        _pages = []
        _queries = []

        for text in nlp.pipe(pages, batch_size=10000, n_threads=16):
            sents = []
            for sent in text.sents:
                sent = [x.text for x in sent if not x.is_space]
                if len(sent) < 2:
                    continue
                if sent[0] == "=" and sent[1] == "=":
                    continue
                sents.append(sent)
            _pages.append(sents)

        for text in nlp.pipe(queries, batch_size=10000, n_threads=16):
            _queries.append(" ".join(x.text for x in text))

        for i in range(len(examples)):
            examples[i][3] = _pages[i]
            examples[i][4] = _queries[i]

        """ split pages into paragraphs"""

        _examples = []
        for i, row in enumerate(tqdm(examples)):
            qid, q_num, a_num, page, query, a_title = row
            paragraph = []
            for sent in page:
                if len(paragraph) <= 50:
                    paragraph += sent
                    continue
                offset_mapping = dict()
                curr = 0
                for j, word in enumerate(paragraph):
                    new = {x: j for x in range(curr, curr + len(word) + 1)}
                    offset_mapping.update(new)
                    curr += len(word) + 1
                paragraph = " ".join(paragraph)
                match = re.search(a_title, paragraph)
                start = end = -1
                if match is not None:
                    start = offset_mapping[match.start()]
                    end = offset_mapping[match.end()]
                _examples.append(
                    BonusPair(qid, q_num, a_num, paragraph, query, a_title, start, end)
                )
                paragraph = []

        with open(save_dir, "w") as f:
            json.dump(_examples, f)

        return _examples


def main():
    dataset = BonusPairsDataset("bonus_pairs.json")


if __name__ == "__main__":
    main()
