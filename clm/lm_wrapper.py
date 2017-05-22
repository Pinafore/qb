from collections import defaultdict
import zlib
import gzip
import time
import ctypes

from nltk.tokenize import RegexpTokenizer
from nltk import bigrams

from qanta import logging
from qanta.util.environment import QB_QUESTION_DB
from qanta.util.constants import CLM_PATH, QB_SOURCE_LOCATION, WIKI_LOCATION
from qanta.config import conf
from qanta.wikipedia.cached_wikipedia import CachedWikipedia
from qanta.datasets.quiz_bowl import QuestionDatabase

from clm import clm
from qanta.util.io import safe_open

log = logging.get(__name__)

kTOKENIZER = RegexpTokenizer('[A-Za-z0-9]+').tokenize

kUNK = "OOV"
kSTART = "STRT"
kEND = "END"
kMAX_TEXT_LENGTH = 5000


def text_iterator(use_wiki, wiki_location,
                  use_qb, qb_location,
                  use_source, source_location,
                  limit=-1,
                  min_pages=0):
    qdb = QuestionDatabase()
    doc_num = 0

    cw = CachedWikipedia()
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
        assert self._vocab_final, "Vocab must be finalized before looking up words"

        # return -1

        # ----------------------------------------------

        if word in self._vocab:
            return self._vocab[word]
        else:
            return self._vocab[kUNK]

    @staticmethod
    def normalize_title(corpus, title):
        norm_title = corpus + title
        return norm_title

    @staticmethod
    def tokenize_without_censor(sentence):
        yield kSTART
        for ii in kTOKENIZER(sentence):
            yield ii.lower()
        yield kEND

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
        yield self.vocab_lookup(kSTART)
        for ii in kTOKENIZER(sentence):
            yield self.vocab_lookup(ii.lower())
        yield self.vocab_lookup(kEND)


class LanguageModelReader(LanguageModelBase):
    def __init__(self, lm_file, interp=0.8, min_span=2, start_rank=200, smooth=0.001, cutoff=-2,
                 slop=0, give_score=False, log_length=True, censor_slop=True, hash_names=False,
                 max_span=5, stopwords=["for", "10", "points", "ftp", "ten", "name"]):

        super().__init__()
        self._loaded_lms = set()
        self._datafile = lm_file
        self._lm = clm.JelinekMercerFeature()
        self._sentence = clm.intArray(kMAX_TEXT_LENGTH)
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
        for file_name in [x for x in self._corpora if x.startswith("compare")]:
            self._loaded_lms.add(self._corpora[file_name])
            self._lm.read_counts("%s/%i" % (self._datafile, self._corpora[file_name]))

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
        for ii, jj in bigrams(tokenized):
            result["wrd"].append(reverse_vocab[ii][:7])
            result["uni_cnt"].append(self._lm.unigram_count(guess_id, ii))
            result["bi_cnt"].append(self._lm.bigram_count(guess_id, ii, jj))

        return result

    def preprocess_and_cache(self, sentence):
        if self._sentence_hash != hash(sentence):
            self._sentence_hash = hash(sentence)
            tokenized = list(self.tokenize_and_censor(sentence))
            self._sentence_length = len(tokenized)
            assert self._sentence_length < kMAX_TEXT_LENGTH
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
    def __init__(self, vocab_size, comparison_corpora):
        super().__init__()
        self._vocab_size = vocab_size
        self._vocab_final = False
        self._vocab = {}
        self._compare = comparison_corpora

        self._training_counts = DistCounter()
        self._obs_counts = {}
        self._sort_voc = None

        # Unigram counts
        self._unigram = defaultdict(DistCounter)

    def train_seen(self, word, count=1):
        """
        Tells the language model that a word has been seen @count times.  This
        will be used to build the final vocabulary.
        """
        assert not self._vocab_final, "Trying to add new words to finalized vocab"

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
            self._vocab = dict((x, y + 1) for y, x in enumerate(vocab))

            assert kUNK not in self._vocab, "Vocab already has %s" % kUNK
            for ii in [kSTART, kEND]:
                assert ii in self._vocab, \
                    "%s missing from %s" % (ii, str(self._vocab.keys()))
            self._vocab[kUNK] = 0
            self._sort_voc = sorted(self._vocab, key=lambda x: self._vocab[x])
        else:
            self._vocab = vocab

        # -------------------------------------------------
        # Add one for the unknown tokens
        del self._training_counts
        return self._vocab

    def add_counts(self, corpus, sentence):

        if corpus not in self._obs_counts:
            self._obs_counts[corpus] = defaultdict(DistCounter)

        # TODO: add start/end tokens (perhaps as option)
        for context, word in bigrams(self.tokenize_and_censor(sentence)):
            self._obs_counts[corpus][context].inc(word)
            self._unigram[corpus].inc(word)

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

        outfile.write("%i\n" % len(self._unigram))
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
        for ii in sorted(self._unigram):
            yield ii

    def write_corpus(self, corpus, offset, outfile):
        num_contexts = len(self._obs_counts[corpus].keys())
        outfile.write("%s %i %i\n" % (corpus, offset, num_contexts))

        for ww in sorted([x for x in self._sort_voc
                          if self._vocab[x] in self._obs_counts[corpus]],
                         key=lambda x:
                         self._obs_counts[corpus][self._vocab[x]].N(),
                         reverse=True):
            ii = self._vocab[ww]
            num_bigrams = self._obs_counts[corpus][ii].B()
            total = self._obs_counts[corpus][ii].N()
            outfile.write("%s %i %i %i\n" %
                          (ww, ii, total, num_bigrams))

            for jj in sorted(self._obs_counts[corpus][ii]):
                assert isinstance(jj, int), "Not an integer: %s" % str(jj)
                assert isinstance(self._obs_counts[corpus][ii][jj], int), \
                    "Got %s for %s %s %s" % (self._obs_counts[corpus][ii][jj], corpus, ii, jj)

                outfile.write("%i %i\n" % (jj, self._obs_counts[corpus][ii][jj]))


def build_clm(lm_out=CLM_PATH, vocab_size=100000, global_lms=5, max_pages=-1):
    min_appearances = conf['clm']['min_appearances']
    log.info("Training language model with pages that appear more than %i times" % min_appearances)

    lm = LanguageModelWriter(vocab_size, global_lms)
    num_docs = 0
    background = defaultdict(int)
    # Initialize language models
    for title, text in text_iterator(True, WIKI_LOCATION,
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
        for title, text in text_iterator(wiki, WIKI_LOCATION,
                                         qb, QB_QUESTION_DB,
                                         source, QB_SOURCE_LOCATION,
                                         max_pages,
                                         min_pages=min_appearances):
            doc_num += 1
            if doc_num % 500 == 0 or time.time() - start > 10:
                log.info("Adding train doc %i, %s (%s)" % (doc_num, title, corpus))
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
