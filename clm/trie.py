# This file should match the API of the C version (ctrie) and
# duplicate behavior exactly.  Goal is for both to be interchangeable,
# but C version to be much more efficient.

from collections import defaultdict
from lm_base import LanguageModelBase

from qanta import logging

log = logging.get(__name__)

class Sentence:
    def __init__(self, max_length):
        self._max = max_length
        self._data = [0] * max_length

    def __setitem__(self, key, val):
        self._data[key] = val

    def __getitem__(self, key):
        assert key < self._max, "%i out of range (%i)" % (key, self._max)
        return self._data[key]

    

class TrieLanguageModel(LanguageModelBase):
    def __init__(self, order=3, start_index=2):
        self._compare = {}
        self._subtotals = {}
        self._contexts = {}
        self._order = order
        self._slop = 0
        self._min_span = 1
        self._start_rank = 200
        self._unigram_smooth = 0.01
        self._cutoff = -2
        self._censor_slop = True
        self._give_score = True
        self._log_length = True
        self._stopwords = set()
        self._start_index = start_index

        self._jm = [1.0 / order] * order

        assert self._start_index <= order, \
            "index (%i) greater than than order (%i)" % (start_index, order)

    def add_count(self, corpus, ngram, count=1):
        assert len(ngram) == self._order, "Count %s wrong length (%i)" % \
          (str(ngram), self._order)

        if corpus not in self._contexts:
            self._contexts[corpus] = {}
            self._subtotals[corpus] = defaultdict(int)
            
        context = self._contexts[corpus]
        last_index = 0
        for ii in range(self._start_index, self._order + 1):
            prefix = ngram[last_index:ii]

            context[prefix] = context.get(prefix, {})
            context = context[prefix]
            last_index = ii

        for ii in range(self._order + 1):
            self._subtotals[corpus][ngram[0:ii]] += count

    def feature(self, corpus, guess, sentence, length):
        return "FOO"
            
    def corpora(self):
        for ii in self._subtotals:
            yield ii
            
    def num_corpora(self):
        return len(self._subtotals)
            
    def count(self, corpus, ngram):
        return self._subtotals[corpus][ngram]

    def total(self, corpus, ngram):
        assert corpus in self._subtotals, "%s not in corpora (%s)" % \
          (str(corpus), str(list(self.corpora()))[:50])
        return self._subtotals[corpus][ngram[:-1]]

    def read_vocab(self, filename):
        """
        This function doesn't actually do anything, but is included to match C API
        """
        None
    
    def jm(self, ngram, theta=None):
        if theta is None:
            theta = self._jm

        # get this context probability
        val = 0
        for nn, ww in zip([ngram[0:x] for x in range(self._order)], theta):
            prob = self.count(nn) / self.total(nn)
            val += ww * prob

    def add_stop(self, word):
        self._stopwords.add(word)
            
    def mle(self, ngram):
        return self.count(ngram) / self.total(ngram)

    def set_compare(self, corpus, compare):
        self._compare[corpus] = compare

    def write_lm(self, corpus_name, corpus_id, outfile):
        """
        Write the LM to a text file
        """

        full_contexts = dict((x, y) for x, y in self._subtotals[corpus_name].items() 
                             if len(x) == self._order)
        outfile.write("%i %i %s\n" % (corpus_id, len(full_contexts), corpus_name))
        lines_written = 0
        for context, count in sorted(full_contexts.items(), reverse=True,
                                     key=lambda x: (x[1], len(x[0]))):
            lines_written += 1
            context_string = " ".join(str(x) for x in context)
            outfile.write("%i\t%s\n" % (count, context_string))
        assert lines_written==len(full_contexts), "Wrong number contexts"
        return lines_written

    def set_jm_interpolation(self, val):
        self._jm = val

    # TODO(jbg): The testing inferface expects to be able to read
    # counts as a string.  However, the C API expects to get a
    # filename.
    def read_counts(self, filename):
        """
        Read the language model from a file
        """

        num_contexts = -1
        with open(filename, 'r') as infile:
            for ii in infile:
                if num_contexts < 0:
                    # Header line
                    corpus = int(ii.split()[0])
                    num_contexts = int(ii.split()[1])
                    assert num_contexts > 0, "Empty language model"
                else:
                    # Every other line
                    fields = [int(x) for x in ii.split()]
                    ngram = tuple(fields[1:])
                    assert len(ngram) == self._order, "Bad line %s" % ii
                    self.add_count(corpus, ngram, fields[0])
        return num_contexts

    def next_word(self, corpus, ngram):
        """
        Traverse the trie ngram and return the final set of contexts
        """

        if len(ngram) < self._start_index:
            return {}
        else:
            context = self._contexts[corpus][ngram[:self._start_index]]
        for ii in ngram[self._start_index:self._order]:
            if ii in context:
                context = context[ii]
            else:
                context = {}
                break
        return context

    def set_slop(self, val):
        self._slop = val

    def set_cutoff(self, val):
        self._cutoff = val

    def set_censor_slop(self, val):
        self._censor_slop = val

    def set_log_length(self, val):
        self._log_length = val

    def set_score(self, val):
        self._score = val

    def set_min_start_rank(self, val):
        self._min_start_rank = val                

    def set_min_span(self, val):
        self._min_span = val

    def set_max_span(self, val):
        self._max_span = val

    def set_unigram_smooth(self, val):
        self._unigram_smooth = val             
                
if __name__ == "__main__":
    tlm = TrieLanguageModel(3)

    tlm.add_count(0, "aab", 2)
    tlm.add_count(0, "aac", 1)
    tlm.add_count(0, "aba", 1)

    print(tlm._contexts)

    print("----------")

    print(tlm.count(0, "aab"))
    print(tlm.total(0, "aab"))
    print(tlm.mle(0, "aab"))
    print(tlm.next_word(0, "a"))
    print(tlm.next_word(0, "aa"))
