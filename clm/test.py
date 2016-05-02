import unittest
from collections import defaultdict

import clm
import lm_wrapper

kCORPUS = [(1, "the name of the rose"),
           (1, "the name of the father"),
           (1, "the name of the father"),
           (1, "the name of the father"),
           (1, "the name of the father"),
           (1, "name of god"),
           (1, "rose water"),
           (0, "rose foo"),
           (2, "tinted water"),
           (2, "tinted god"),
           (2, "tinted father"),
           (2, "rose tinted glasses"),
           (2, "rose tinted glasses")]

kVOCAB = ["OOV", "the", "name", "of", "father", "tinted", "rose",
          "water", "god"]

kBIGRAM = {0: {"rose": ["OOV"]},
           1: {"the": ["name"] * 5 + ["father"] * 4 + ["rose"] + ["god"],
               "name": ["of"] * 6, "of": ["god"] + ["the"] * 5},
           2: {"rose": ["tinted"] * 2, "tinted": ["water", "god", "father"] +
               ["glasses"] * 2}}
# Add in comparison corpora


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
        self._compare = 2
        self._wrap = lm_wrapper.LanguageModelWriter(len(kVOCAB) - 1,
                                                    self._compare)
        self._lm = clm.JelinekMercerFeature()

        for cc, ss in kCORPUS:
            for tt in self._wrap.tokenize_without_censor(ss):
                self._wrap.train_seen(tt)
        self._wrap.finalize()

        for cc, ss in kCORPUS:
            self._wrap.add_train("toy", str(cc), ss)
            # print(cc, ss, self._wrap._unigram.keys())

    def test_counts(self):
        sw = StubWriter()
        self._wrap.write_lm(sw)

        self.assertEqual(len(set(x[0] for x in kCORPUS)) +
                         self._compare, int(sw[0]))
        self.assertEqual(len(kVOCAB), int(sw[1]))

        line = 2
        for ww in kVOCAB:
            self.assertEqual(ww, sw[line])
            line += 1

        for ii in xrange(self._compare):
            self.assertEqual("compare_%i %i" %
                             (ii, hash("compare_%i" % ii) % self._compare),
                             sw[line])
            line += 1

        print("Corpora check %i" % line)
        print(list(sorted(self._wrap._unigram)))
        for ii in xrange(3):
            corpus = "toy%i" % ii
            self.assertEqual("%s %i" % (corpus, hash(ii) % self._compare),
                             sw[line])
            line += 1

        for ii in xrange(3):
            for jj, ww in enumerate(kVOCAB):
                print(ii, jj, ww)
                if ww in kBIGRAM[ii]:
                    self.assertEqual("%i %i %i" %
                                     (jj, len(kBIGRAM[ii][ww]),
                                      len(set(kBIGRAM[ii][ww]))), sw[line])
                else:
                    self.assertEqual("%i 0 0" % jj, sw[line])

                line += 1



if __name__ == '__main__':
    unittest.main()
