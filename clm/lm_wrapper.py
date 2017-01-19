# TODO:
#
# Make sure runs with existing python trie
# Add in training for a guesser
# Add in demo for guessing
# C++ implementation
# Implement guesser interface

from collections import defaultdict
import time
import re
import os
import ctypes

from unidecode import unidecode

from nltk.tokenize import RegexpTokenizer
from nltk import ngrams

from qanta import logging
from qanta.util.build_whoosh import text_iterator
from qanta.util.environment import QB_QUESTION_DB, QB_WIKI_LOCATION
from qanta.util.constants import CLM_PATH, QB_SOURCE_LOCATION, QB_STOP_WORDS, \
    MIN_APPEARANCES, CLM_ORDER, CLM_COMPARE, CLM_VOCAB, \
    CLM_CUTOFF, CLM_SLOP, CLM_GIVE_SCORE, CLM_LOG_LENGTH, CLM_CENSOR_SLOP, \
    CLM_MAX_LENGTH, CLM_MIN_SPAN, CLM_START_RANK, CLM_UNK_TOK, CLM_START_TOK, \
    CLM_END_TOK, CLM_HASH_NAMES, CLM_MAX_SPAN

from clm import ctrie
from clm.trie import TrieLanguageModel
from clm.lm_base import LanguageModelBase
from qanta.util.io import safe_open

log = logging.get(__name__)

kTOKENIZER = RegexpTokenizer('[A-Za-z0-9]+').tokenize
kLEFT_PAD = '<s>'

kGOODCHAR = re.compile(r"[a-zA-Z0-9]*")


def pretty_debug(name, result, max_width=10):
    if not result:
        return ""

    length = max(len(result[x]) for x in result)
    display = defaultdict(str)
    display[0] = name

    start_row = 1
    for ii in range(length):
        if ii % max_width == 0:
            if ii > 0:
                start_row += len(result)
                display[start_row] = ""
                start_row += 1

            for jj, val in enumerate(sorted(result)):
                display[start_row + jj] = "%s\t" % val

        for jj, val in enumerate(sorted(result)):
            display[start_row + jj] += "%s\t" % str(result[val][ii])

    return "\n".join(display[x] for x in sorted(display))


class DistCounter:
    def __init__(self):
        self._total = 0
        self._bins = 0
        self._counts = defaultdict(int)

    def __len__(self):
        return len(self._counts)

    def inc(self, word, count=1):
        self._counts[word] += count
        self._total += count
        assert self._total >= 0, "Total cannot go negative"
        assert self._counts[word] >= 0, "Element count cannot go negative"
        if self._counts[word] == 0:
            del self._counts[word]

    def __getitem__(self, word):
        return self._counts.get(word, 0)

    def __iter__(self):
        for ii in self._counts:
            if self._counts[ii] > 0:
                yield ii

    def B(self):
        return len(self._counts)

    def N(self):
        return self._total

    def freq(self, word):
        if self._total == 0:
            return 0
        return self._counts.get(word, 0) / float(self._total)

class LanguageModelReader(LanguageModelBase):
    def __init__(self, lm_file, interp=0.8, smooth=0.001,
                 min_span=CLM_MIN_SPAN, start_rank=CLM_START_RANK,
                 cutoff=CLM_CUTOFF, slop=CLM_SLOP,
                 give_score=CLM_GIVE_SCORE, log_length=CLM_LOG_LENGTH,
                 censor_slop=CLM_CENSOR_SLOP,
                 hash_names=CLM_HASH_NAMES, max_span=CLM_MAX_SPAN,
                 stopwords=QB_STOP_WORDS):

        super().__init__()
        self._loaded_lms = set()
        self._datafile = lm_file
        self._lm = ctrie.JelinekMercerFeature()
        self._sentence = ctrie.intArray(CLM_MAX_LENGTH)
        self._sentence_length = 0
        self._sentence_hash = 0
        self._vocab_final = True
        self._hash_names = hash_names
        self._stopwords = stopwords
        self._vocab = {}
        self._corpora = {}
        self._sort_voc = None

        assert(max_span >= min_span), "Max span %i must be greater than min %i" % \
            (max_span, min_span)

        self.set_params(interp, min(min_span, max_span), max(min_span, max_span),
                        start_rank, smooth, cutoff, slop, censor_slop, give_score,
                        log_length, stopwords)

    def set_params(self, interp, min_span, max_span, start_rank, smooth,
                   cutoff, slop, censor_slop, give_score,
                   log_length, stopwords):
        assert isinstance(min_span, int), "Got bad span %s" % str(min_span)
        self._lm.set_interpolation(interp)
        self._lm.set_slop(slop)
        self._lm.set_cutoff(cutoff)
        self._lm.set_min_span(min_span)
        self._lm.set_max_span(max_span)
        self._lm.set_smooth(smooth)
        self._lm.set_min_start_rank(start_rank)
        self._lm.set_score(give_score)
        self._lm.set_log_length(log_length)
        self._lm.set_censor_slop(censor_slop)
        self._stopwords = stopwords

    def init(self):
        infile = open("%s.txt" % self._datafile)
        num_lms = int(infile.readline())

        vocab_size = int(infile.readline())
        for i in range(vocab_size):
            self._vocab[infile.readline().strip()] = i
        if vocab_size > 100:
            log.info("Done reading %i vocab (Python)" % vocab_size)

        for i in range(num_lms):
            line = infile.readline()
            log.info("LINE {}: {}".format(i, line))
            corpus, compare = line.split()
            self._corpora[corpus] = i

        corpora_keys = list(self._corpora.keys())
        if len(corpora_keys) > 10:
            log.info(corpora_keys[:10])

        self._lm.read_vocab("%s.txt" % self._datafile)

        # Add stop words that are in vocabulary (unknown word is 0)
        for i in [self.vocab_lookup(x) for x in self._stopwords if self.vocab_lookup(x) != 0]:
            self._lm.add_stop(i)

        # Load comparisons language model
        for i in [x for x in self._corpora if x.startswith("compare")]:
            self._loaded_lms.add(self._corpora[i])
            self._lm.read_counts("%s/%i" % (self._datafile, self._corpora[i]))

    def verbose_feature(self, corpus, guess, sentence):
        """
        Debug what's going on
        """

        result = defaultdict(list)
        reverse_vocab = dict((y, x) for x, y in self._vocab.items())

        tokenized = list(self.tokenize_and_censor(sentence))
        norm_title = self.normalize_title(corpus, guess)
        if norm_title not in self._corpora:
            return result
        guess_id = self._corpora[norm_title]

        # Get the counts of words in unigram and bigram
        for ii, jj in ngrams(tokenized, CLM_ORDER):
            result["wrd"].append(reverse_vocab[ii][:7])
            result["uni_cnt"].append(self._lm.unigram_count(guess_id, ii))
            result["bi_cnt"].append(self._lm.bigram_count(guess_id, ii, jj))

        return result

    def feature_line(self, corpus, guess, sentence):
        if self._sentence_hash != hash(sentence):
            self._sentence_hash = hash(sentence)
            tokenized = list(self.tokenize_and_censor(sentence))
            self._sentence_length = len(tokenized)
            assert self._sentence_length < kMAX_TEXT_LENGTH
            for ii, ww in enumerate(tokenized):
                self._sentence[ii] = ww

        norm_title = self.normalize_title(corpus, guess)
        if norm_title not in self._corpora or self._sentence_length == 0:
            return ""
        else:
            guess_id = self._corpora[norm_title]
            if guess_id not in self._loaded_lms:
                self._lm.read_counts("%s/%i" % (self._datafile, guess_id))
                self._loaded_lms.add(guess_id)

            feat = self._lm.feature(corpus, guess_id, self._sentence, self._sentence_length)

            if self._hash_names:
                result = []
                for ii in feat.split():
                    if "_" in ii:
                        if ":" in ii:
                            name, val = ii.split(":")
                            hashed_name = ctypes.c_uint32(hash(name)).value
                            result.append("%i:%s" % (hashed_name, val))
                        else:
                            result.append("%i" % ctypes.c_uint32(hash(ii)).value)
                    else:
                        result.append(ii)
                return " ".join(result)
            else:
                return feat


class LanguageModelWriter(LanguageModelBase):
    def __init__(self, vocab_size, comparison_corpora, order=2):
        super().__init__()
        self._vocab_size = vocab_size
        self._vocab_final = False
        self._vocab = {}
        self._compare = comparison_corpora

        self._order = order
        self._training_counts = DistCounter()
        self._lms = {}
        self._sort_voc = None

        # Unigram counts

    def train_seen(self, word, count=1):
        """
        Tells the language model that a word has been seen @count times.  This
        will be used to build the final vocabulary.
        """
        assert not self._vocab_final, \
            "Trying to add new words to finalized vocab"

        self._training_counts.inc(word, count)

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

    def add_counts(self, corpus, sentence):

        if corpus not in self._obs_counts:
            self._lms[corpus] = TrieLanguageModel(self._order, 1)

        # TODO: add start/end tokens (perhaps as option)
        for ii in ngrams(self.tokenize_and_censor(sentence)):
            self._lms

    def add_train(self, corpus, title, sentence):
        """
        Add the counts associated with a sentence.
        """
        norm_title = self.normalize_title(corpus, title)
        comp = self.compare(norm_title)

        self.add_counts(norm_title, sentence)
        for ii in range(self._compare):
            if comp != ii:
                self.add_counts("compare_%i" % ii, sentence)

    def compare(self, title):
        return hash(title) % self._compare

    def write_vocab(self, outfile):
        """
        Write the text-based language model to a file
        """

        # TODO(jbg): actually write the correct mean and variance

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
                log.info("{} {}".format(cc, self.compare(cc)))

            corpus_num += 1

    def corpora(self):
        for ii in sorted(self._lms):
            yield ii


def build_clm(lm_out=CLM_PATH, vocab_size=CLM_VOCAB, global_lms=CLM_COMPARE,
              max_pages=-1):
    log.info("Training LM with pages appearing %i times" % MIN_APPEARANCES)

    lm = LanguageModelWriter(vocab_size, global_lms)
    num_docs = 0
    background = defaultdict(int)
    # Initialize language models
    for title, text in text_iterator(True, QB_WIKI_LOCATION,
                                     True, QB_QUESTION_DB,
                                     True, QB_SOURCE_LOCATION,
                                     max_pages,
                                     min_pages=MIN_APPEARANCES):
        num_docs += 1
        if num_docs % 500 == 0:
            log.info("{} {}".format(unidecode(title), num_docs))
            log.info(str(list(lm.tokenize_without_censor(text[100:200]))))

        for tt in lm.tokenize_without_censor(text):
            background[tt] += 1

    # Create the vocabulary
    for ii in background:
        lm.train_seen(ii, background[ii])
    vocab = lm.finalize()
    log.info(str(vocab)[:80])
    log.info("Vocab size is {} from {} docs".format(len(vocab), num_docs))
    del background

    # Train the language model
    doc_num = 0
    for corpus, qb, wiki, source in [("wiki", False, True, False),
                                     ("qb", True, False, False),
                                     ("source", False, False, True)
                                     ]:
        # Add training data
        start = time.time()
        for title, text in text_iterator(wiki, QB_WIKI_LOCATION,
                                         qb, QB_QUESTION_DB,
                                         source, QB_SOURCE_LOCATION,
                                         max_pages,
                                         min_pages=MIN_APPEARANCES):
            doc_num += 1
            if doc_num % 500 == 0 or time.time() - start > 10:
                log.info("Adding train doc %i, %s (%s)" % (doc_num, unidecode(title), corpus))
                start = time.time()
            lm.add_train(corpus, title, text)

    log.info("Done training")
    if lm_out:
        # Create the extractor object and write out the pickle
        with safe_open("%s.txt" % lm_out, 'w') as f:
            lm.write_vocab(f)

        for ii, cc in enumerate(lm.corpora()):
            with safe_open("%s/%i" % (lm_out, ii), 'w') as f:
                lm.write_corpus(cc, ii, f)


if __name__ == "__main__":
    build_clm(max_pages=100)
