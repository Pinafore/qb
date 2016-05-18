# Code to test languge model feature extractor. Unit test seems to hang (at
# least on OS X), seems to be related to reading in counts.  Don't know
# solution.

import unittest
from collections import defaultdict
import os
import shutil
from math import log, exp

from nltk import bigrams

import clm
import lm_wrapper
from lm_wrapper import kUNK, kSTART, kEND

kCORPUS = [("toy1", "the name of the rose"),
           ("toy1", "the name of the father"),
           ("toy1", "the name of the father"),
           ("toy1", "the name of the father"),
           ("toy1", "the name of the father"),
           ("toy1", "name of god"),
           ("toy1", "rose water"),
           ("toy0", "rose foo"),
           ("toy2", "tinted water"),
           ("toy2", "tinted god"),
           ("toy2", "tinted father"),
           ("toy2", "rose tinted glasses"),
           ("toy2", "rose tinted glasses")]
kCOMPARE = 2

kVOCAB = ["OOV", kEND, kSTART, "the", "name", "of", "father", "rose", "tinted",
          "glasses", "god", "water"]

kSMOOTH = [0.1, 1.0, 1000]
kINTERP = [0.1, 0.5, 0.75, 1.0]

kQUERIES = [[kSTART] + x.split() + [kEND] for x in ["the name of the rose",
                                                    "the name of the name",
                                                    "name of the god",
                                                    "tinted rose",
                                                    "foo foo foo"]]


def filter_feat(term):
    return not ("LEN" in term or "MAX" in term or "PROB" in term)

# TODO(jbg): remove low frequency words
# Compute bigrams
kBIGRAM = {}
for cc, ss in kCORPUS:
    if not cc in kBIGRAM:
        kBIGRAM[cc] = defaultdict(list)
    words = [x if x in kVOCAB else kVOCAB[0] for x in
             [kSTART] + ss.split() + [kEND]]
    for ii, ww in enumerate(words):
        if ii > 0:
            kBIGRAM[cc][words[ii - 1]].append(ww)

for ii in set(x[0] for x in kCORPUS):
    comp = "compare_%i" % (hash(ii) % kCOMPARE)
    for jj in xrange(kCOMPARE):
        # warning: only works with single digit corpora
        if not comp.endswith(str(jj)):
            if not "compare_%i" % jj in kBIGRAM:
                kBIGRAM["compare_%i" % jj] = defaultdict(list)
            for kk in kBIGRAM[ii]:
                kBIGRAM["compare_%i" % jj][kk] += kBIGRAM[ii][kk]


# Add in comparison corpora
def match_slop(matches, slop):
    slop_match = [False] * len(matches)
    for start in range(1, len(matches)):
        for end in range(start + 1, len(matches)):
            if not all(matches[x] for x in range(start, end)):
                num_matches = sum(1 for x in range(start, end) if matches[x])
                if num_matches > 0 and num_matches + slop >= end - start:
                    for kk in range(start, end):
                        slop_match[kk] = not matches[kk]
    for ii in range(len(matches)):
        if matches[ii]:
            slop_match[ii] = True
        elif slop_match[ii]:
            slop_match[ii] = "SLOP"
        else:
            slop_match[ii] = False

    return slop_match


def matcher(input, counts, guess, slop=0, always=[], never=[]):
    matches = [True] + \
        [False] * (len(input) - 1)
    index = 1
    for aa, bb in bigrams(x if x in kVOCAB else kUNK for x in input):
        if bb in counts[guess][aa] or bb in always:
            matches[index] = True
        index += 1
    if slop > 0:
        matches = match_slop(matches, slop)
    return matches


def gen_line(input, matches, vocab, guess, corpus,
             min_span, log_len, censor_slop=True):
    max_length = 0
    tokens = input
    result = []

    for ii, start in enumerate(matches):
        for jj, end in enumerate(matches):
            if all(matches[x] for x in xrange(ii, jj + 1)):
                max_length = max(max_length,
                                 jj - ii + 1 if
                                 (jj > ii or (jj == ii and ii > 0)) else 0)
                if jj - ii >= min_span:
                    match = guess
                    for kk in xrange(ii, jj + 1):
                        if matches[kk] == "SLOP" and censor_slop:
                            match += "_SLOP"
                        else:
                            match += "_%s" % tokens[kk]
                    result.append(match)
    if log_len:
        result.append("%s_LGLEN:%0.2f" % (corpus, log(1 + max_length)))
    result.append("%s_LEN:%i" % (corpus, max_length))
    return result


def score_span(lm, corpus, compare, vocab, span):
    tokens = span.split("_")[1:]

    components = {}
    components["d-norm"] = lm.unigram_norm(corpus)
    components["b-norm"] = lm.unigram_norm(compare)
    components["smooth"] = lm.smooth()
    components["0-start-%s-d" % tokens[0]] = \
        lm.score(corpus, -1, kVOCAB.index(tokens[0]))
    components["0-start-%s-b" % tokens[0]] = \
        -lm.score(compare, -1, kVOCAB.index(tokens[0]))
    components["0-start-%s-dc" % tokens[0]] = \
        lm.unigram_count(corpus, kVOCAB.index(tokens[0]))
    components["0-start-%s-bc" % tokens[0]] = \
        lm.unigram_count(compare, kVOCAB.index(tokens[0]))

    index = 0
    for aa, bb in bigrams(kVOCAB.index(x) for x in tokens):
        index += 1
        key = (index, kVOCAB[aa], kVOCAB[bb])
        components["%i-%s-%s-d" % key] = lm.score(corpus, aa, bb)
        components["%i-%s-%s-b" % key] = -lm.score(compare, aa, bb)
        components["%i-%s-%s-dc" % key] = lm.bigram_count(corpus, aa, bb)
        components["%i-%s-%s-bc" % key] = lm.bigram_count(compare, aa, bb)

    return sum(y for x, y in components.iteritems()
               if x.endswith("-d") or x.endswith("-b")), \
        components


class StubWriter:
    """
    Class to check the output of writing LM counts.
    """

    def __init__(self):
        self._cache = []

    def write(self, line):
        self._cache.append(line.strip())

    def __getitem__(self, item):
        return self._cache[item]

    def __iter__(self):
        for ii in self._cache:
            yield ii


class TestStringMethods(unittest.TestCase):
    def setUp(self):
        self._compare = kCOMPARE
        self._writer = lm_wrapper.LanguageModelWriter(len(kVOCAB) - 1,
                                                      self._compare)
        self._lm = clm.JelinekMercerFeature()

        for cc, ss in kCORPUS:
            for tt in self._writer.tokenize_without_censor(ss):
                self._writer.train_seen(tt)
        self._writer.finalize()

        for cc, ss in kCORPUS:
            self._writer.add_train("toy", cc.replace("toy", ""), ss)

        o = open("temp_toy_lm.txt", 'w')
        self._writer.write_vocab(o)
        o.close()

        if not os.path.exists("temp_toy_lm"):
            os.makedirs("temp_toy_lm")

        for ii, cc in enumerate(self.corpora()):
            o = open("temp_toy_lm/%i" % ii, 'w')
            self._writer.write_corpus(cc, ii, o)
            o.close()

        self._lm.read_vocab("temp_toy_lm.txt")
        for ii, cc in enumerate(self.corpora()):
            self._lm.read_counts("temp_toy_lm/%i" % ii)

        self._reader = lm_wrapper.LanguageModelReader("temp_toy_lm")
        self._reader.init()

    def tearDown(self):
        None
        #os.remove("temp_toy_lm.txt")
        #shutil.rmtree("temp_toy_lm")

    def test_vocab(self):
        sw = StubWriter()
        self._writer.write_vocab(sw)

        self.assertEqual(len(set(x[0] for x in kCORPUS)) +
                         self._compare, int(sw[0]))
        self.assertEqual(len(kVOCAB), int(sw[1]))

        line = 2
        for ww in kVOCAB:
            self.assertEqual(ww, sw[line])
            line += 1

        for ii in self.corpora():
            self.assertEqual("%s %i" %
                             (ii, hash(ii) % self._compare),
                             sw[line])
            line += 1

    def test_corpora_counts(self):
        for ii, cc in enumerate(self.corpora()):
            sw = StubWriter()

            num_contexts = sum(1 for x in kBIGRAM[cc]
                               if len(kBIGRAM[cc][x]) > 0)
            self._writer.write_corpus(cc, ii, sw)
            self.assertEqual("%s %i %i" % (cc, ii, num_contexts),
                             sw[0])
            line = 1

            for ww in sorted([x for x in kVOCAB if len(kBIGRAM[cc][x]) > 0],
                             key=lambda x: len(kBIGRAM[cc][x]), reverse=True):
                jj = kVOCAB.index(ww)

                # Check unigram counts
                self.assertEqual("%s %i %i %i" %
                                 (ww, jj, len(kBIGRAM[cc][ww]),
                                  len(set(kBIGRAM[cc][ww]))), sw[line])
                line += 1
                # then check bigram counts
                for kk in sorted(set(kVOCAB.index(x) for x in
                                     kBIGRAM[cc][ww])):
                    expected = "%i %i" % \
                        (kk, sum(1 for x in
                                 kBIGRAM[cc][ww] if x == kVOCAB[kk]))
                    msg = "%s -> %s in %s: " % (ww, kVOCAB[kk], ii)
                    msg += "%s (ex) vs %s (actual)" % (expected, sw[line])
                    self.assertEqual(expected, sw[line], msg)
                    line += 1

    def test_unigram_counts(self):
        normalizer = {}
        for jj, ww in enumerate(kVOCAB):
            for ii, cc in enumerate(self.corpora()):
                if not cc in normalizer:
                    normalizer[cc] = sum(len(kBIGRAM[cc][ww]) for
                                         ww in kBIGRAM[cc])
                if ww in kBIGRAM[cc]:
                    count = len(kBIGRAM[cc][ww])
                    self.assertEqual(count,
                                     self._lm.unigram_count(ii, jj))

                    for ss in kSMOOTH:
                        self._lm.set_smooth(ss)
                        denom = (normalizer[cc] + ss * len(kVOCAB))
                        self.assertAlmostEqual(denom,
                                               self._lm.unigram_norm(ii),
                                               places=5)
                        p = log((float(count) + ss) / denom)
                        self.assertAlmostEqual(p, self._lm.score(ii, -1, jj),
                                               places=5)

    def test_unknown(self):
        self.assertEqual(self._reader.vocab_lookup(lm_wrapper.kUNK), 0)

    def corpora(self):
        return ["compare_%i" % x for x in range(self._compare)] + \
            ["toy%i" % x for x in xrange(3)]

    def test_bigram_counts(self):
        for ii, start in enumerate(kVOCAB):
            for jj, end in enumerate(kVOCAB):
                for kk, cc in enumerate(self.corpora()):
                    count = sum(1 for x in kBIGRAM[cc][start] if x == end)
                    self.assertEqual(count, self._lm.bigram_count(kk, ii, jj),
                                     ("%s (%i) -> %s (%i) [%s, %i]: " %
                                      (start, ii, end, jj, cc, kk)) +
                                     ("%i (exp) vs. %i (got)" %
                                      (count,
                                       self._lm.bigram_count(kk, ii, jj))))

                    for ss in kSMOOTH:
                        self._lm.set_smooth(ss)
                        for mm in kINTERP:
                            self._lm.set_interpolation(mm)
                            uni_sum = sum(len(kBIGRAM[cc][ww]) for
                                          ww in kBIGRAM[cc])
                            uni_num = len(kBIGRAM[cc][end]) + ss
                            uni_den = self._lm.unigram_norm(kk)
                            uni = uni_num / float(uni_den)

                            if len(kBIGRAM[cc][start]) > 0:
                                bi_num = self._lm.bigram_count(kk, ii, jj)
                                bi_den = self._lm.unigram_count(kk, ii)
                                bi = float(bi_num) / float(bi_den)
                            else:
                                bi_num = 0
                                bi_den = 0
                                bi = 0.0

                            p = log(mm * uni + (1 - mm) * bi)

                            msg = "Bigram %s (%i) -> %s (%i):" % \
                                (start, ii, end, jj)
                            msg += "\n\tUNI (%i + %f)/(%i + %f)" % \
                                (len(kBIGRAM[cc][end]), ss, uni_sum,
                                 ss * len(kVOCAB))
                            msg += "= %f / %f" % \
                                (self._lm.unigram_count(kk, jj) +
                                 self._lm.smooth(), uni_den)
                            msg += "= %f " % uni

                            msg += "\n\tBI %i / %i = %f" % (bi_num, bi_den, bi)

                            msg += "\n\t%f * %f + %f * %f = %f" % (mm, uni,
                                                                   1 - mm, bi,
                                                                   exp(p))

                            val = self._lm.score(kk, ii, jj)
                            msg += "\n\tgot: %f (%f)" % (val, exp(val))

                            self.assertAlmostEqual(p, val, msg=msg, places=5)

    def test_unconstrained_lines(self):
        test = 0
        corpus = ""
        for span in [1, 2, 3, 5]:
            for slop in [0, 1, 2, 3]:
                self._reader.set_params(.5, span, 0, 1.0, -1e6, slop, True,
                                        False, True, [])
                for qq in kQUERIES:
                    for guess in ["toy0", "toy1", "toy2"]:
                        test += 1
                        matches = matcher(qq, kBIGRAM, guess, slop=slop)
                        exp = gen_line(qq, matches, kVOCAB, guess, "", span,
                                       True)
                        act = self._reader.feature_line(corpus, guess,
                                                        " ".join(qq[1:-1]))
                        context = {"test": test, "span": span, "query": qq,
                                   "guess": guess, "slop": matches,
                                   "expected": " ".join(sorted(exp)),
                                   "actual": act,
                                   "matches": matcher(qq, kBIGRAM, guess),
                                   "slop val": slop}
                        for ii, jj in zip(sorted(exp), sorted(act.split())):
                            message = "\t%s\t%s\n" % (ii, jj) + \
                                "\n".join("%s\t%s" % (x, y) for x, y in
                                          context.iteritems())
                            self.assertEqual(ii, jj, message)

    def test_z_span_scores(self):
        corpora = self.corpora()
        for ss in kSMOOTH:
            for mm in kINTERP:
                for span_length in [1, 2, 3, 5]:
                    self._reader.set_params(mm, span_length, 0, ss, -1e6,
                                            1, False, True, False, [])
                    self._lm.set_smooth(ss)
                    self._lm.set_interpolation(mm)
                    for qq in kQUERIES:
                        for guess in ["toy0", "toy1", "toy2"]:
                            comp = "compare_%i" % (hash(guess) % kCOMPARE)
                            val = self._reader.feature_line("", guess,
                                                            " ".join(qq[1:-1]))
                            for span, score in [x.split(":") for x in
                                                val.split() if filter_feat(x)]:
                                exp, cc = score_span(self._lm,
                                                     corpora.index(guess),
                                                     corpora.index(comp),
                                                     kVOCAB, span)
                                message = "span:%s\n" % span
                                message += "exp:%f\n" % exp
                                message += "act:%f\n" % float(score)
                                message += "query:%s\n" % " ".join(qq)
                                for aa, bb in sorted(cc.iteritems()):
                                    message += "\t%s:%f\n" % (aa, bb)

                                self.assertAlmostEqual(exp, float(score),
                                                       msg=message, places=1)


if __name__ == '__main__':
    unittest.main()
