# TODO:
#
# Make sure runs with existing python trie
# Add in training for a guesser
# Add in demo for guessing
# C++ implementation
# KN scoring
# Implement guesser interface

from collections import defaultdict
import zlib
import gzip
import time
import ctypes

from unidecode import unidecode
from nltk import ngrams, bigrams

from qanta import logging
from qanta.util.environment import QB_QUESTION_DB, QB_WIKI_LOCATION
from qanta.util.constants import CLM_PATH, QB_SOURCE_LOCATION, QB_STOP_WORDS, \
    MIN_APPEARANCES, CLM_ORDER, CLM_COMPARE, CLM_VOCAB, \
    CLM_CUTOFF, CLM_SLOP, CLM_GIVE_SCORE, CLM_LOG_LENGTH, CLM_CENSOR_SLOP, \
    CLM_MAX_LENGTH, CLM_MIN_SPAN, CLM_START_RANK, CLM_UNK_TOK, CLM_START_TOK, \
    CLM_END_TOK, CLM_HASH_NAMES, CLM_MAX_SPAN, CLM_USE_C_VERSION

if CLM_USE_C_VERSION:
    from clm import ctrie
    from ctrie import JelinekMercerFeature as TrieLanguageModel
    from ctrie import intArray as Sentence
else:
    from trie import TrieLanguageModel
    from trie import Sentence

from lm_base import LanguageModelBase
from qanta.util.constants import CLM_PATH, QB_SOURCE_LOCATION
from qanta.config import conf
from qanta.preprocess import format_guess
from qanta.wikipedia.cached_wikipedia import CachedWikipedia
from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.util.environment import data_path
from qanta.util.constants import COUNTRY_LIST_PATH

from clm import clm
from qanta.util.io import safe_open

log = logging.get(__name__)

kLEFT_PAD = '<s>'

kMAX_TEXT_LENGTH = 5000


def text_iterator(use_wiki, wiki_location,
                  use_qb, qb_location,
                  use_source, source_location,
                  limit=-1,
                  min_pages=0, country_list=COUNTRY_LIST_PATH):
    qdb = QuestionDatabase()
    doc_num = 0

    cw = CachedWikipedia(wiki_location, data_path(country_list))
    pages = qdb.questions_with_pages()

    for p in sorted(pages, key=lambda k: len(pages[k]), reverse=True):
        # This bit of code needs to line up with the logic in qdb.py
        # to have the same logic as the page_by_count function
        if len(pages[p]) < min_pages:
            continue

        if use_qb:
            train_questions = [x for x in pages[p] if x.fold == "train"]
            question_text = "\n".join(" ".join(x.raw_words()) for x in train_questions)
        else:
            question_text = ''

        if use_source:
            filename = '%s/%s' % (source_location, p)
            if os.path.isfile(filename):
                try:
                    with gzip.open(filename, 'rb') as f:
                        source_text = f.read()
                except zlib.error:
                    log.info("Error reading %s" % filename)
                    source_text = ''
            else:
                source_text = ''
        else:
            source_text = u''

        if use_wiki:
            wikipedia_text = cw[p].content
        else:
            wikipedia_text = u""

        total_text = wikipedia_text
        total_text += "\n"
        total_text += question_text
        total_text += "\n"
        total_text += str(source_text)

        yield p, total_text
        doc_num += 1

        if 0 < limit < doc_num:
            break


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
        self._datafile = lm_file
        self._lm = TrieLanguageModel()
        self._sentence = Sentence(CLM_MAX_LENGTH)
        self._sentence_length = 0
        self._sentence_hash = 0
        self._vocab_final = True
        self._hash_names = hash_names
        self._stopwords = stopwords
        self._sort_voc = None
        self._loaded_lms = set()

        assert(max_span >= min_span), "Max span %i must be greater than min %i" % \
            (max_span, min_span)

        self.set_params(interp, min(min_span, max_span), max(min_span, max_span),
                        start_rank, smooth, cutoff, slop, censor_slop, give_score,
                        log_length, stopwords)

    def set_params(self, interp, min_span, max_span, start_rank, smooth,
                   cutoff, slop, censor_slop, give_score,
                   log_length, stopwords):
        assert isinstance(min_span, int), "Got bad span %s" % str(min_span)
        self._lm.set_jm_interpolation(interp)
        self._lm.set_slop(slop)
        self._lm.set_cutoff(cutoff)
        self._lm.set_min_span(min_span)
        self._lm.set_max_span(max_span)
        self._lm.set_unigram_smooth(smooth)
        self._lm.set_min_start_rank(start_rank)
        self._lm.set_score(give_score)
        self._lm.set_log_length(log_length)
        self._lm.set_censor_slop(censor_slop)
        self._stopwords = stopwords

    def init(self):
        self._read_vocab_and_corpora("%s.txt" % self._datafile)

        # Load comparisons language model
        for ii in [x for x in self._corpora if x.startswith("compare_")]:
            self._loaded_lms.add(self._corpora[ii])
            filename = "%s/%i" % (self._datafile, self._corpora[ii])
            log.info("reading %s" % filename)
            self._lm.read_counts(filename)

    def verbose_feature(self, corpus, guess, sentence):
        """
        Debug what's going on
        """

        result = defaultdict(list)
        reverse_vocab = dict((y, x) for x, y in self._vocab.items())

        tokenized = list(self.tokenize_and_censor(sentence, pad=True))
        norm_title = self.normalize_title(corpus, guess)
        if norm_title not in self._corpora:
            return result
        guess_id = self._corpora[norm_title]

        # Get the counts of words in unigram and bigram
        for ii in ngrams(tokenized, CLM_ORDER):
            result["wrd"].append(" ".join(reverse_vocab[x] for x in ii))
            result["uni_cnt"].append(self._lm.total(guess_id, ii))
            result["bi_cnt"].append(self._lm.count(guess_id, ii))

        return result

    def preprocess_and_cache(self, sentence):
        if self._sentence_hash != hash(sentence):
            self._sentence_hash = hash(sentence)
            tokenized = list(self.tokenize_and_censor(sentence, pad=True))
            self._sentence_length = len(tokenized)
            assert self._sentence_length < CLM_MAX_LENGTH
            for ii, ww in enumerate(tokenized):
                self._sentence[ii] = ww

    def dict_feat(self, corpus, guess, sentence):
        """
        Return a dictionary of the features
        """

        self.preprocess_and_cache(sentence)
        guess_id = self._corpora[norm_title]
        if guess_id not in self._loaded_lms:
            self._lm.read_counts("%s/%i" % (self._datafile, guess_id))
            self._loaded_lms.add(guess_id)

        feat = self._lm.feature(corpus, guess_id, self._sentence, self._sentence_length)

        d = {}
        for ii in feat.split():
            if ":" in ii:
                key, val = ii.split(":")
            else:
                key = ii
                val = 1
            d[key] = val
        return d

    def feature_line(self, corpus, guess, sentence):
        self.preprocess_and_cache(sentence)

        norm_title = self.normalize_title(corpus, guess)
        if norm_title not in self._corpora or self._sentence_length == 0:
            return ""
        else:
            guess_id = self._corpora[norm_title]
            if guess_id not in self._loaded_lms:
                filename = "%s/%i" % (self._datafile, guess_id)

                contexts = self._lm.read_counts(filename)
                log.info("read %s (%i contexts)" % (filename, contexts))
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
    def __init__(self, vocab_size, comparison_corpora, order):
        super().__init__()
        self._vocab_size = vocab_size
        self._vocab_final = False
        self._vocab = {}
        self._compare = comparison_corpora

        self._order = order
        self._training_counts = DistCounter()
        self._lm = TrieLanguageModel()

        # Unigram counts

    def train_seen(self, word, count=1):
        """
        Tells the language model that a word has been seen @count times.  This
        will be used to build the final vocabulary.
        """
        assert not self._vocab_final, \
            "Trying to add new words to finalized vocab"

        self._training_counts.inc(word, count)

    def add_counts(self, corpus, sentence):

        # TODO: add start/end tokens (perhaps as option)
        for ii in ngrams(self.tokenize_and_censor(sentence, pad=False),
                         self._order, pad_left=True, pad_right=True,
                         left_pad_symbol=self.vocab_lookup(CLM_START_TOK),
                         right_pad_symbol=self.vocab_lookup(CLM_END_TOK)):
            self._lm.add_count(corpus, ii)

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
        if title.startswith("compare_"):
            return -1
        else:
            return hash(title) % self._compare

    def corpora(self):
        for ii in sorted(self._unigram):
            yield ii

    def write_corpus(self, corpus_name, id, file):
        return self._lm.write_lm(corpus_name, id, file)

    def num_corpora(self):
        return self._lm.num_corpora()

    def corpora(self):
        for ii in self._lm.corpora():
            yield ii


def build_clm(lm_out=CLM_PATH, vocab_size=CLM_VOCAB, global_lms=CLM_COMPARE,
              max_pages=-1):
    log.info("Training LM with pages appearing %i times" % MIN_APPEARANCES)
    min_appearances = conf['clm']['min_appearances']
    log.info("Training language model with pages that appear more than %i times" % min_appearances)

    lm = LanguageModelWriter(vocab_size, global_lms, CLM_ORDER)
    num_docs = 0
    background = defaultdict(int)
    # Initialize language models
    for title, text in text_iterator(True, QB_WIKI_LOCATION,
                                     True, QB_QUESTION_DB,
                                     True, QB_SOURCE_LOCATION,
                                     max_pages,
                                     min_pages=min_appearances):
        num_docs += 1
        if num_docs % 500 == 0:
            log.info("{} {}".format(title, num_docs))
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
                                         min_pages=min_appearances):
            doc_num += 1
            if doc_num % 500 == 0 or time.time() - start > 10:
                log.info("Adding train doc %i, %s (%s)" %
                         (doc_num, unidecode(title), corpus))
                start = time.time()
            lm.add_train(corpus, title, text)

    log.info("Done training")
    if lm_out:
        # Create the extractor object and write out the pickle
        with safe_open("%s.txt" % lm_out, 'w') as f:
            lm.write_vocab(f)

        for ii, cc in enumerate(lm.corpora()):
            with safe_open("%s/%i" % (lm_out, ii), 'w') as f:
                if ii % 100 == 0:
                    log.info("Write LM corpus %s to %s" %
                             (cc, "%s/%i" % (lm_out, ii)))
                lm.write_corpus(cc, ii, f)


if __name__ == "__main__":
    build_clm(max_pages=5)
