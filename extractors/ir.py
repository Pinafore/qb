# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

from future.builtins import range, str
import six
from string import ascii_lowercase, ascii_uppercase, digits
from collections import defaultdict

from numpy import isnan

import whoosh
from whoosh import index
from whoosh import qparser
from whoosh import scoring
from whoosh.collectors import TimeLimitCollector, TimeLimit

from unidecode import unidecode

from extractors.abstract import FeatureExtractor
from util.constants import (
    NEG_INF, STOP_WORDS, PAREN_EXPRESSION, get_treebank_tokenizer, get_punctuation_table)
from util.environment import data_path

QUERY_CHARS = set(ascii_lowercase + ascii_uppercase + digits)

tokenizer = get_treebank_tokenizer()
valid_strings = set(ascii_lowercase) | set(str(x) for x in range(10)) | set(' ')


class IrIndex(object):
    def __init__(self, location, mean, var, num_results, time_limit):
        self._name = location
        self._index = index.open_dir(location, readonly=True)
        self._query_hash = 0
        self._query_terms = None

        self._document_cache = {}

        self._limit = num_results
        self._time = time_limit

        # TODO(jbg): This is a parameter that can be optimized
        og = qparser.OrGroup.factory(0.9)
        self._text_parser = qparser.QueryParser("content", self._index.schema, group=og)
        self._id_parser = qparser.QueryParser("id", self._index.schema, group=og)

        if not isnan(mean):
            self._mean = mean
        else:
            self._mean = 0.0
        if not isnan(var):
            self._var = var
        else:
            self._var = 1.0

        self._var = var
        self._misses = 0
        self._hits = 0

    def scale(self, val):
        return (val - self._mean) / self._var

    # @profile
    def score_one_guess(self, title, text):
        # Return impossible score if there's no text
        if isinstance(text, list):
            text = " ".join(text)
        if not text:
            return NEG_INF

        # Get the document
        if self._query_hash != hash(text):
            # print("New query for %s" % text)
            text_query, text_length = self.create_query(text)
            try:
                self._query_terms = [tuple(x.terms()) for x in text_query]
                self._query_terms = [x[0] for x in self._query_terms]
            except NotImplementedError:
                # TODO: This branch happens when there's only one content word
                # in the query.  Not sure it's being handled correctly.
                # print(text_query)
                self._query_terms = [("content", text_query)]
            self._query_hash = hash(text)

            # Put the query words into the IDF lookup
            self._idf = {}
            with self._index.searcher(weighting=scoring.TF_IDF()) as r:
                for field, word in self._query_terms:
                    try:
                        self._idf[word] = r.idf(field, word)
                    except ValueError:
                        # print("Error computing idf for %s" % word)
                        self._idf[word] = 0.0
                    except TypeError:
                        terms = list(word.terms())
                        # print(terms)
                        if len(terms) > 0:
                            word = terms[0][1]
                            self._idf[word] = r.idf(field, word)
                            # print(word, self._idf[word])

        score = 0.0
        # Two cases: we need the searcher to create the document vector, in
        # which case it doesn't hurt to have it around for idf computations
        # --- OR ---
        # We don't need the searcher around, so we perhaps save on some IDF
        # lookups.
        if title not in self._document_cache:
            self._misses += 1
            # print("Cache miss for %s %f %s" % (unidecode(title),
            #                                    float(self._hits) /
            #                                    (self._misses + self._hits),
            #                                    self._name))
            with self._index.searcher(weighting=scoring.TF_IDF()) as r:
                docnum = r.document_number(id=title)
                if docnum is None:
                    self._document_cache[title] = None
                    return NEG_INF
                vec = dict(r.vector_as("frequency", docnum, "content"))
                self._document_cache[title] = vec
        else:
            self._hits += 1
            # print("Cache hit for %s %f %s" % (unidecode(title),
            #                                   float(self._hits) /
            #                                   (self._misses + self._hits),
            #                                   self._name))
            vec = self._document_cache[title]
            if vec is None:
                return NEG_INF

        length = 0
        for field, word in [x for x in self._query_terms if x[1] in vec]:
            length += 1
            score += self._idf[word] * vec[word]
        if length > 0:
            score /= float(length)

        # TODO: My implementation of idf isn't matching *exactly* with theirs,
        # but it should be much faster, and it's close

        # print("My way %f" % score)

        # allow_q = query.Term("id", title)
        # backup_score = kNEG_INF
        # with self._index.searcher(weighting=scoring.TF_IDF()) as s:
        #     text_query, text_length = self.create_query(text)
        #     res = s.search(text_query, limit=1, filter=allow_q)
        #     if not res.is_empty():
        #         backup_score = res[0].score
        # print("Their way %f" % backup_score)

        return self.scale(score)

    @staticmethod
    def normalize(text):
        """
        Normalize text (but leave punctuation; c.f. prepare_query)
        """
        text = PAREN_EXPRESSION.sub("", text)
        text = unidecode(text).lower()
        text = " ".join(x for x in text.split() if x not in STOP_WORDS)
        return ''.join(x for x in text if x in valid_strings)

    @staticmethod
    def prepare_query(raw_text):
        """
        Given the raw text of a query, filter out inadmissable characters
        """
        punctuation_table = get_punctuation_table()
        search_tokens = [x.translate(punctuation_table)
                         for x in tokenizer(str(raw_text))
                         if len(x) > 3]
        search_string = u" ".join(
                ''.join(filter(lambda y: y in QUERY_CHARS, unidecode(x))) for x in search_tokens
        )
        return search_string, len(search_tokens)

    def create_query(self, raw_text, edit_dist=0, title=False):
        # TODO: Much of this processesing is from an abundance of
        # caution in squashing a bug.  It can likely be simplified and
        # sped up.
        if isinstance(raw_text, list):
            raw_text = u" ".join(raw_text)

        # print(search_string)
        search_string, search_len = self.prepare_query(raw_text)
        search_query = self._text_parser.parse(search_string)

        return search_query, search_len

    def full_search(self, query, time_limit=-1, search_limit=50, edit_dist=0):
        val = {}

        with self._index.searcher(weighting=scoring.TF_IDF()) as searcher:
            if time_limit > 0:
                c = searcher.collector(limit=search_limit)
                tlc = TimeLimitCollector(c, timelimit=time_limit)
                try:
                    searcher.search_with_collector(query, tlc)
                except TimeLimit:
                    pass
                try:
                    res = tlc.results()
                except TimeLimit:
                    res = []
            else:
                res = searcher.search(query, limit=search_limit)

            for ii in res:
                val[ii['title']] = (ii.docnum, self.scale(ii.score))
        return val

    def text_guess(self, text):
        text_query, text_length = self.create_query(text)

        results = {}
        try:
            for kk, vv in six.iteritems(self.full_search(
                    text_query, time_limit=self._time, search_limit=self._limit)):
                results[unidecode(kk)] = vv[1]
        except whoosh.searching.TimeLimit:
            pass
        return results


class IrExtractor(FeatureExtractor):
    def __init__(self, k_min_appearances, num_results=50, time_limit=0.05):
        super(IrExtractor, self).__init__()
        self._limit = num_results
        self._time = time_limit
        self.name = "ir"
        self._index = {}
        self.k_min_appearances = k_min_appearances

    def set_metadata(self, answer, category, qnum, sent, token, guesses, fold):
        super(IrExtractor, self).set_metadata(answer, category, qnum, sent, token, guesses, fold)

        self.add_index("wiki_%i" % self.k_min_appearances,
                       "%s_%i" % (data_path("data/ir/whoosh_wiki"), self.k_min_appearances))
        self.add_index("qb_%i" % self.k_min_appearances,
                       "%s_%i" % (data_path("data/ir/whoosh_qb"), self.k_min_appearances))
        self.add_index("source%i" % self.k_min_appearances,
                       "%s_%i" % (data_path("data/ir/whoosh_source"), self.k_min_appearances))

    def add_index(self, name, location, mean=0.0, variance=1.0):
        if name not in self._index:
            print("Adding %s (%s)" % (location, name))
            self._index[name] = IrIndex(location, mean, variance,
                                        self._limit, self._time)
            print("Current set of indices: %s (%f, %f)" %
                  (list(self._index.keys()), mean, variance))

    @staticmethod
    def has_guess():
        return False

    def score_one_guess(self, title, text):
        val = {}
        for ii in self._index:
            val[ii] = self._index[ii].score_one_guess(title, text)
        return val

    def vw_from_title(self, title, text):
        val = self.score_one_guess(title, text)
        return self.vw_from_score(val)

    def vw_from_score(self, results):
        res = "|%s" % self.name
        for ii in results:
            if results[ii] > NEG_INF:
                res += " %sfound:1 %sscore:%f" % \
                    (ii, ii, self._index[ii].scale(results[ii]))
            else:
                res += " %sfound:0 %sscore:0.0" % (ii, ii)
        return res

    def text_guess(self, text):
        res = defaultdict(dict)
        to_delete = set()
        for ii in self._index:
            feat_guess = self._index[ii].text_guess(text)
            #            try:
            #    feat_guess = self._index[ii].text_guess(text)
            # except zlib.error:
            #    print("Error reading index %s" % ii)
            #    to_delete.add(ii)
            for jj in feat_guess:
                res[jj][ii] = feat_guess[jj]
        if to_delete:
            for ii in to_delete:
                del self._index[ii]

        # Symmeterize the guesses
        for ii in res:
            for jj in [x for x in self._index if x not in res[ii]]:
                res[ii][jj] = self._index[jj].score_one_guess(ii, text)
        return res

    def features(self, question, candidate):
        pass

    def guesses(self, question):
        pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Demo for IR guesser")
    parser.add_argument("--whoosh_qb", default=data_path("data/ir/whoosh_wiki_4"),
                        help="Location of IR index of qb")
    parser.add_argument("--whoosh_wiki", default=data_path("data/ir/whoosh_qb_4"),
                        help="Location of IR index of wiki")

    flags = parser.parse_args()

    import time

    start = time.time()

    ws = IrExtractor()
    ws.add_index("qb", flags.whoosh_qb, 100, 50)
    ws.add_index("wiki", flags.whoosh_wiki, 100, 50)

    print("Startup: %f sec" % (time.time() - start))

    tests = {
        u"Tannhäuser (opera)": u"""He sought out the pope to
    seek forgiveness of his sins, only to be told that just as the pope's staff
    would never (*) blossom, his sins are never be forgiven. Three days later,
    the pope's staff miraculously bore flowers. For 10 points--identify this
    German folk hero, the subject of an opera by Wagner [VAHG-ner].""",
        u"Transformers: The Movie": u"""This movie wasn't a masterpiece of
    writing, featuring such lines as "spare me this mockery of justice"
    and "yes, friends and now destroy Unicron (*), kill
    the grand poobah, eliminate even the toughest stains."  It even lifted
    phrases such as "Klaatu Barada Nikto" from reputable sources, but if the
    dialog didn't insult your intelligence, the film also featured the musical
    talents of Weird Al with "Dare to be Stupid."  This was, however, a
    commercial success, and was not lacking in star power, with Casey Casem as
    Cliffjumper, Leonard Nimoy as Galvatron, Eric Idle as Wreck-Gar, and Orson
    Welles in his last role as Unicron the giant robot planet.  For ten points,
    identify this 1986 animated film pitting the Autobots against the
    Decepticons, based on the successful cartoon and toy franchise which was
    "More than meets the eye."."""}

    guesses = ["Arkansas", "Australia", "Transformers", "Aaron Burr"]

    for ii in tests:
        print(ii)
        # start = time.time()
        # res = dict(ws.text_guess(tests[ii]))
        # elapsed = time.time() - start
        # print("%f secs for %s: %s" % (elapsed, ii, str(res)))

        for gg in guesses:
            print("Score for %s: %s" %
                  (gg, str(ws.score_one_guess(gg, tests[ii]))))
