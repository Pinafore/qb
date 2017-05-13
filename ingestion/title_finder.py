from itertools import chain
from collections import defaultdict
from string import ascii_lowercase, ascii_uppercase

import qanta
from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.wikipedia.cached_wikipedia import CachedWikipedia
from qanta import logging
from ingestion.page_assigner import PageAssigner

from fuzzywuzzy import process
from fuzzywuzzy.fuzz import UWRatio

log = logging.get(__name__)


def scorer(left, right):
    if right.startswith("list of") or \
       right.endswith(" topics") or \
       right.startswith("wikiproject"):
        val = 0
    else:
        val = UWRatio(left, right)
    return val


def reasonable_case(page):
    """
    Checks that a wikipedia page doesn't have crazy capitalization
    (which often leads to bad matches.
    """

    return len(page) > 2 and page[0] in ascii_uppercase and \
        all(x in ascii_lowercase for x in page[1:])


class TitleFinder:
    def __init__(self, index, wiki, known_pages,
                 normalize=lambda x: x, prune=1500):
        import gzip

        self.normalize = normalize
        self._index = defaultdict(set)
        self._wiki = wiki
        self._prune = set()
        self._known = known_pages

        # map single words to the relevant wikipedia titles
        with gzip.open(index) as f:
            line = 0
            for title in f:
                line += 1
                if line == 1:
                    continue

                converted_title = None
                for word in [normalize(x) for x in
                             title.decode('utf-8').split("_") if len(x) > 2]:
                    if converted_title is None:
                        converted_title = title.decode('utf-8').strip()
                    if len(word) > 2:
                        self._index[word].add(converted_title)
                if line % 5000000 == 0:
                    log.info("%i %s: %s -> %s" % (line, title, word, list(self._index[word])[:3]))
                    self.prune(prune)
        self.prune(prune)

        # Take another pass just to add exact titles
        with gzip.open(index) as f:
            for ii in f:
                title = ii.decode('utf-8').strip()
                if "(" not in title and reasonable_case(title):
                    norm = normalize(title.replace("_", "_"))
                    self._index[norm].add(title)

    def prune(self, prune):
        self._prune |= set(x for x in self._index
                           if len(self._index[x]) > prune)
        log.info("Pruning %s" % str(list(self._prune)[:50]))
        for ii in self._prune:
            if ii in self._index:
                del self._index[ii]

    def query(self, text):
        norm = self.normalize(text)
        tokens = norm.split()
        candidates = set(chain.from_iterable(self._index[x] for x in tokens
                                             if x in self._index))

        # try looking for plurals
        if tokens[-1].endswith("s") and tokens[-1][:-1] in self._index:
            candidates |= self._index[tokens[-1][:-1]]

        # try looking for exact match
        if norm in self._index:
            candidates |= self._index[norm]

        candidates = dict((self.normalize(x.replace("_", " ")), x) for x in
                          candidates)

        return candidates

    def score(self, text, score_function=scorer):
        candidates = self.query(text)

        candidates = process.extract(text, candidates, limit=len(candidates),
                                     scorer=scorer)

        collapsed = defaultdict(int)
        for wiki, val, norm in candidates:
            page = self._wiki.redirect(wiki)
            if scorer(self.normalize(text),
                      self.normalize(page.replace("_", " "))) != 0:
                collapsed[page] += val

            # Give bonus to exact matches
            if self.normalize(page) == self.normalize(text):
                collapsed[page] += 250

            if page in self._known:
                collapsed[page] += 250

        return collapsed

    def best_guess(self, unassigned, min_val=50, delta=5):
        results = {}
        guess_num = 0
        for ii in [x for x in unassigned if len(x) > 2]:
            v = self.score(ii)
            if len(v) >= 2:
                scores = sorted(v, key=v.get, reverse=True)
                top = v[scores[0]]
                second = v[scores[1]]

                if top - second >= delta and top > min_val:
                    results[ii] = scores[0]
                guess_num += 1

                if guess_num % 1000 == 0:
                    log.info("Matching %s -> %s" % (ii, results.get(ii, None)))
            elif len(v) == 1:
                if max(v.values()) > min_val:
                    results[ii] = max(v.keys())

        return results


if __name__ == "__main__":
    import argparse
    from glob import glob
    parser = argparse.ArgumentParser(description='Import questions')
    parser.add_argument('--direct_path', type=str,
                        default='data/internal/page_assignment/direct/')
    parser.add_argument('--ambiguous_path', type=str,
                        default='data/internal/page_assignment/ambiguous/')
    parser.add_argument('--unambiguous_path', type=str,
                        default='data/internal/page_assignment/unambiguous/')
    flags = parser.parse_args()

    pa = PageAssigner(QuestionDatabase.normalize_answer)
    for ii in glob("%s/*" % flags.ambiguous_path):
        pa.load_ambiguous(ii)
    for ii in glob("%s/*" % flags.unambiguous_path):
        pa.load_unambiguous(ii)
    for ii in glob("%s/*" % flags.direct_path):
        pa.load_direct(ii)

    cw = CachedWikipedia()
    tf = TitleFinder("data/enwiki-latest-all-titles-in-ns0.gz", cw,
                     pa.known_pages(),
                     normalize=QuestionDatabase.normalize_answer)



    for ii in ['die leiden des jungen werthers', '99 Luftballons', 'saint nicholas of myra', 'édouard roche', 'the mahdi or mohammad ahmed', 'the first vatican council', 'antietam national battlefield', 'cia', 'samuel f b morse', 'the passion according to st matthew or st matthew’s passion or matthäuspassion', 'another world', 'rolling in the deep', 'tony gwynn', 'opal', 'tylenol', 'queues', 'dachau', 'lipoproteins', 'haiku', 'japan', 'zoroastrianism']:
        A = tf.score(ii)
        print("--------")
        num = 0
        for ii in sorted(A, key=A.get, reverse=True):
            num += 1
            print("\t%s\t%i" % (ii, A[ii]))

            if num > 10:
                break
