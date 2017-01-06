
from collections import defaultdict

class TrieLanguageModel:
    def __init__(self, order = 3, start_index = 2):
        self._subtotals = defaultdict(int)
        self._contexts = {}
        self._order = order
        self._start_index = start_index
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

    def kn(self, ngram, theta):
        # get this context probability
        counts = self.count(ngram)
        
    
    def mle(self, ngram):
        return self.count(ngram) / self.total(ngram)
            
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
