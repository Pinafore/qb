import re

from unidecode import unidecode
from nltk.tokenize import RegexpTokenizer

from qanta import logging
from qanta.util.constants import CLM_START_TOK, CLM_END_TOK, CLM_UNK_TOK

kGOODCHAR = re.compile(r"[a-zA-Z0-9]*")
kTOKENIZER = RegexpTokenizer('[A-Za-z0-9]+').tokenize

log = logging.get(__name__)


class LanguageModelBase:
    def __init__(self, order):
        self._vocab_final = None
        self._vocab = None
        self._stopwords = set()
        self._corpora = {}
        self._sort_voc = None
        self._order = order

    def write_vocab(self, outfile):
        """
        Write the text-based language model to a file
        """

        # TODO(jbg): actually write the correct mean and variance

        outfile.write("%i\n" % self.num_corpora())
        outfile.write("%i\n" % len(self._vocab))
        vocab_size = len(self._vocab)
        for ii in self._sort_voc:
            outfile.write("%s\n" % ii)
        if vocab_size > 100:
            log.info("Done writing vocab")

        corpus_num = 0
        for cc in self.corpora():
            outfile.write("%s %i\n" % (cc, self.compare(cc)))

            if corpus_num % 100 == 0:
                log.info("Corpus compare write {} {}".format(cc,
                                                             self.compare(cc)))

            corpus_num += 1

    def finalize(self, vocab=None):
        """
        Fixes the vocabulary as static, prevents keeping additional vocab from
        being added
        """
        self._vocab_final = True
        if vocab is None:
            self._vocab_size = min(len(self._training_counts),
                                   self._vocab_size)
            vocab = sorted(self._training_counts)
            vocab = sorted(vocab, key=lambda x: self._training_counts[x],
                           reverse=True)[:self._vocab_size]

            # Add three for unk, start, and end
            self._vocab = dict((x, y + 3) for y, x in enumerate(vocab))

            for vv, ii in enumerate([CLM_UNK_TOK, CLM_START_TOK, CLM_END_TOK]):
                assert ii not in self._vocab, \
                    "%s already in from %s" % (ii, str(self._vocab.keys()))
                self._vocab[ii] = vv

            self._sort_voc = sorted(self._vocab, key=lambda x: self._vocab[x])
        else:
            self._vocab = vocab

        # -------------------------------------------------
        # Add one for the unknown tokens
        del self._training_counts
        return self._vocab

    def read_vocab_and_corpora(self, filename):
        log.info("Opening %s for LM input" % filename)
        infile = open(filename)
        num_lms = int(infile.readline())

        self._vocab = {}
        vocab_size = int(infile.readline())
        for ii in range(vocab_size):
            self._vocab[infile.readline().strip()] = ii
        if vocab_size > 100:
            log.info("Done reading %i vocab (Python)" % vocab_size)

        for ii in range(num_lms):
            line = infile.readline()
            corpus, compare = line.split()
            compare = int(compare)
            if ii % 1000 == 0:
                log.info("Corpus (%s) loaded: compare %i, line %i" %
                         (corpus, compare, ii))
            self._corpora[corpus] = ii
            self._lm.set_compare(ii, compare)

        corpora_keys = list(self._corpora.keys())
        if len(corpora_keys) > 10:
            log.info(corpora_keys[:10])

        self._lm.read_vocab("%s.txt" % self._datafile)

        # Add stop words that are in vocabulary (unknown word is 0)
        for i in [self.vocab_lookup(x) for x in self._stopwords if self.vocab_lookup(x) != 0]:
            self._lm.add_stop(i)

        self._vocab_final = True

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

    def tokenize_and_censor(self, sentence, pad=False):
        """
        Given a sentence, yields a sentence suitable for training or testing.
        Prefix the sentence with <s>, replace words not in the vocabulary with
        <UNK>, and end the sentence with </s>.
        """
        if not isinstance(sentence, str):
            sentence = ' '.join(list(sentence))
        if pad:
            yield self.vocab_lookup(CLM_START_TOK)
        for ii in kTOKENIZER(unidecode(sentence)):
            yield self.vocab_lookup(ii.lower())
        if pad:
            yield self.vocab_lookup(CLM_END_TOK)
