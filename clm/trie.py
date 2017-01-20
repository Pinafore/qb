from collections import defaultdict
from clm.lm_base import LanguageModelBase


class TrieLanguageModel(LanguageModelBase):
    def __init__(self, order=3, start_index=2):
        self._subtotals = defaultdict(int)
        self._contexts = {}
        self._order = order
        self._start_index = start_index

        self._jm = [1.0 / order] * order

        assert self._start_index <= order, \
            "index (%i) greater than than order (%i)" % (start_index, order)

    def add_count(self, ngram, count=1):
        assert len(ngram) == self._order

        context = self._contexts
        last_index = 0
        for ii in range(self._start_index, self._order + 1):
            prefix = ngram[last_index:ii]

            context[prefix] = context.get(prefix, {})
            context = context[prefix]
            last_index = ii

        for ii in range(self._order + 1):
            self._subtotals[ngram[0:ii]] += count

    def count(self, ngram):
        return self._subtotals[ngram]

    def total(self, ngram):
        return self._subtotals[ngram[:-1]]

    def jm(self, ngram, theta=None):
        if theta is None:
            theta = self._jm

        # get this context probability
        val = 0
        for nn, ww in zip([ngram[0:x] for x in range(self._order)], theta):
            prob = self.count(nn) / self.total(nn)
            val += ww * prob

    def mle(self, ngram):
        return self.count(ngram) / self.total(ngram)

    def write_lm(self, id, outfile):
        """
        Write the LM to a text file
        """

        number_contexts = len(self._contexts)
        outfile.write("%i %i\n" % (id, number_contexts))

        for context, count in sorted(self._subtotals.items(), reverse=True,
                                     key=lambda x: (x[1], len(x[0]))):
            num_tokens = len(context)
            context_string = " ".join(str(x) for x in context)
            outfile.write("%i %i\t%s\n" % (count, num_tokens, context_string))

    def load_lm(self, infile):
        """
        Read the language model from a file
        """

        num_contexts = -1
        for ii in infile:
            if num_contexts < 0:
                num_contexts = int(ii)
                assert num_contexts > 0, "Empty language model"
            fields = ii.split()
            ngram = fields[2:]
            assert len(ngram) == int(ii[1]), "Bad line %s" % ii
            self.add_count(ngram, int(ii[0]))

    def next_word(self, ngram):
        """
        Traverse the trie ngram and return the final set of contexts
        """

        if len(ngram) < self._start_index:
            return {}
        else:
            context = self._contexts[ngram[:self._start_index]]
        for ii in ngram[self._start_index:self._order]:
            if ii in context:
                context = context[ii]
            else:
                context = {}
                break
        return context


if __name__ == "__main__":
    tlm = TrieLanguageModel(3)

    tlm.add_count("aab", 2)
    tlm.add_count("aac", 1)
    tlm.add_count("aba", 1)

    print(tlm._contexts)

    print("----------")

    print(tlm.count("aab"))
    print(tlm.total("aab"))
    print(tlm.mle("aab"))
    print(tlm.next_word("a"))
    print(tlm.next_word("aa"))
