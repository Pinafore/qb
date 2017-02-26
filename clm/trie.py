# This file should match the API of the C version (ctrie) and
# duplicate behavior exactly.  Goal is for both to be interchangeable,
# but C version to be much more efficient.

from collections import defaultdict
from lm_base import LanguageModelBase
from nltk import ngrams

from qanta import logging
from qanta.util.constants import CLM_INT_WORDS

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
        self._score_method = "jm"
        self._censor_slop = True
        self._give_score = True
        self._log_length = True
        self._stopwords = set()
        self._start_index = start_index
        self._vocab = {}

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

    def feature(self, name, corpus, sentence, length):
        compare = self._compare[corpus]
        
        span_length = 0
        num_spans = 0
        max_length = 0
        max_prob = float("-inf")

        # Set counters for keeping track of spans
        start = 0
        history = []
        slop_count = self._slop
        # end of counters for keeping track of spans
        
        index = 0
        res = []
        for ngram in ngrams(sentence, self._order):
             # trim history if it's too long
             if len(history) > self._max_span:
                 history = history[1:]

             # output the history if we've seen enough
             if len(history) > self._min_span:
                 if CLM_INT_WORDS:
                    tokens = "_".join(str(x) for x in history)
                 else:
                    tokens = "_".join(self._vocab[x] for x in history)
                    
                 if self._score:
                     score = self.log_ratio(corpus, compare, history)
                     res.append("%s:%0.2f" % (tokens, score))
                 else:
                    res.append("_".join(str(x) for x in history))
                 
             seen = self.count(corpus, ngram)

             if seen > 0 and len(history) == 0:
                 num_spans += 1
                 span_length += 1
                 start = index
                 history = list(ngram)
             elif seen > 0 and len(history) > 0:
                 span_length += 1
                 history.append(ngram[-1])
             elif seen == 0 and len(history) > 0 and slop_count > 0:
                 slop_count -= 1
                 if self._censor_slop:
                     history.append(QB_LM_SLOP)
                 else:
                     history.append(ngram[-1])
             elif seen == 0 and slop_count == 0:
                 max_length = max(index - start, max_length)
                 history = []
                 slop_count = self._slop

             # Increment the index for keeping track of longest span
             index += 1

        if self._score:
            res.append("%s-PROB:%f" % (name, self.log_ratio(corpus, compare, sentence)))
            res.append("%s-MAXPROB:%f" % (name, max_prob))
            
        res.append("%s-LEN:%i" % (name, max_length))
        res.append("%s-SPAN:%i" % (name, span_length / float(length)))
        res.append("%s-HITS:%i" % (name, num_spans))
        if self._log_length:
            res.append("%s-LGLEN:%f" % (name, log(max_length)))
        return " ".join(res)
            
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
        with open(filename) as infile:
            num_corpora = infile.readline()
            num_words = infile.readline()

            for ii in range(int(num_words)):
                self._vocab[ii] = infile.readline().strip()
    
    def jm(self, corpus, ngram, theta=None):
        if theta is None:
            theta = self._jm

        # get this context probability
        val = 0
        for nn, ww in zip([ngram[0:x] for x in range(self._order)], theta):
            prob = self.count(corpus, nn) / self.total(corpus, nn)
            val += ww * prob

    def log_ratio(self, sequence, corpus, compare):
        val = 0.0
        for ii in ngrams(sequence, self._order):
            if self._score_method == "jm":
                val += self.jm(corpus, ii)
                val -= self.jm(compare, ii)
            elif self._score_method == "kn":
                val += self.kn(corpus, ii)
                val -= self.kn(compare, ii)
        return val
    
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

<<<<<<< HEAD
        number_contexts = len(self._contexts[corpus_name])
        outfile.write("%i %i\n" % (corpus_id, number_contexts))

        full_contexts = dict((x, y) for x, y in self._subtotals[corpus_name].items()
=======
        full_contexts = dict((x, y) for x, y in self._subtotals[corpus_name].items() 
>>>>>>> ee229d8628a54e5f80852a5fe4dea0f5fb2da4eb
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
<<<<<<< HEAD
                    ngram = fields[1:]
                    log.info(str(fields) + str(ngram))
                    assert len(ngram) == fields[1], "Bad line %s" % ii
=======
                    ngram = tuple(fields[1:])
                    assert len(ngram) == self._order, "Bad line %s" % ii
>>>>>>> ee229d8628a54e5f80852a5fe4dea0f5fb2da4eb
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
