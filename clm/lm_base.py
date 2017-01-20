import re

from unidecode import unidecode
from nltk.tokenize import RegexpTokenizer

from qanta.util.constants import CLM_START_TOK, CLM_END_TOK, CLM_UNK_TOK

kGOODCHAR = re.compile(r"[a-zA-Z0-9]*")
kTOKENIZER = RegexpTokenizer('[A-Za-z0-9]+').tokenize


class LanguageModelBase:
    def __init__(self):
        self._vocab_final = None
        self._vocab = None

    def vocab_lookup(self, word):
        """
        Given a word, provides a vocabulary representation.  Words under the
        cutoff threshold shold have the same value.  All words with counts
        greater than or equal to the cutoff should be unique and consistent.
        """
        assert self._vocab_final, "Vocab must be finalized before lookup"

        # return -1

        # ----------------------------------------------

        if word in self._vocab:
            return self._vocab[word]
        else:
            return self._vocab[CLM_UNK_TOK]

    @staticmethod
    def normalize_title(corpus, title):
        norm_title = corpus + "".join(x for x in kGOODCHAR.findall(title) if x)
        return norm_title

    @staticmethod
    def tokenize_without_censor(sentence):
        for ii in kTOKENIZER(unidecode(sentence)):
            yield ii.lower()

    def vocab_size(self):
        return len(self._vocab)

    def tokenize_and_censor(self, sentence):
        """
        Given a sentence, yields a sentence suitable for training or testing.
        Prefix the sentence with <s>, replace words not in the vocabulary with
        <UNK>, and end the sentence with </s>.
        """
        if not isinstance(sentence, str):
            sentence = ' '.join(list(sentence))
        yield self.vocab_lookup(CLM_START_TOK)
        for ii in kTOKENIZER(unidecode(sentence)):
            yield self.vocab_lookup(ii.lower())
        yield self.vocab_lookup(CLM_END_TOK)
