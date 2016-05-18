from collections import defaultdict
import time
import re
import os

from unidecode import unidecode

from nltk.tokenize import RegexpTokenizer
from nltk import bigrams

from util.build_whoosh import text_iterator

import clm

kTOKENIZER = RegexpTokenizer('[A-Za-z0-9]+').tokenize

kUNK = "OOV"
kSTART = "STRT"
kEND = "END"
kMAX_TEXT_LENGTH = 5000
kGOODCHAR = re.compile(r"[a-zA-Z0-9]*")


def pretty_debug(name, result, max_width=10):
    if not result:
        return ""

    length = max(len(result[x]) for x in result)
    display = defaultdict(str)
    display[0] = name

    start_row = 1
    for ii in xrange(length):
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
    def vocab_lookup(self, word):
        """
        Given a word, provides a vocabulary representation.  Words under the
        cutoff threshold shold have the same value.  All words with counts
        greater than or equal to the cutoff should be unique and consistent.
        """
        assert self._vocab_final, \
            "Vocab must be finalized before looking up words"

        # return -1

        # ----------------------------------------------

        if word in self._vocab:
            return self._vocab[word]
        else:
            return self._vocab[kUNK]

    @staticmethod
    def normalize_title(corpus, title):
        norm_title = corpus + "".join(x for x in kGOODCHAR.findall(title) if x)
        return norm_title

    @staticmethod
    def tokenize_without_censor(sentence):
        yield kSTART
        for ii in kTOKENIZER(unidecode(sentence)):
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
        if not isinstance(sentence, basestring):
            sentence = ' '.join(list(sentence))
        yield self.vocab_lookup(kSTART)
        for ii in kTOKENIZER(unidecode(sentence)):
            yield self.vocab_lookup(ii.lower())
        yield self.vocab_lookup(kEND)


class LanguageModelReader(LanguageModelBase):
    def __init__(self, lm_file, interp=0.8, min_span=1, start_rank=200,
                 smooth=0.001, cutoff=-2, slop=0, give_score=True,
                 log_length=True, censor_slop=True,
                 stopwords=["for", "10", "points", "ftp", "ten", "name"]):
        from clm import intArray

        self._loaded_lms = set()
        self._datafile = lm_file
        self._lm = clm.JelinekMercerFeature()
        self._sentence = intArray(kMAX_TEXT_LENGTH)
        self._sentence_length = 0
        self._sentence_hash = 0
        self._vocab_final = True

        self.set_params(interp, min_span, start_rank, smooth, cutoff, slop,
                        censor_slop, give_score, log_length, stopwords)

    def set_params(self, interp, min_span, start_rank, smooth,
                   cutoff, slop, censor_slop, give_score,
                   log_length, stopwords):
        assert isinstance(min_span, int), "Got bad span %s" % str(min_span)
        self._lm.set_interpolation(interp)
        self._lm.set_slop(slop)
        self._lm.set_cutoff(cutoff)
        self._lm.set_min_span(min_span)
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
        self._vocab = {}
        for ii in xrange(vocab_size):
            self._vocab[infile.readline().strip()] = ii
        if vocab_size > 100:
            print("Done reading %i vocab (Python)" % vocab_size)

        self._corpora = {}
        for ii in xrange(num_lms):
            line = infile.readline()
            corpus, compare = line.split()
            self._corpora[corpus] = ii

        if len(self._corpora.keys()) > 10:
            print(self._corpora.keys()[:10])

        self._lm.read_vocab("%s.txt" % self._datafile)

        # Add stop words that are in vocabulary (unknown word is 0)
        for ii in [self.vocab_lookup(x) for x in self._stopwords
                   if self.vocab_lookup(x) != 0]:
            self._lm.add_stop(ii)

        # Load comparisons language model
        for ii in [x for x in self._corpora if x.startswith("compare")]:
            self._loaded_lms.add(self._corpora[ii])
            self._lm.read_counts("%s/%i" % (self._datafile, self._corpora[ii]))

    def verbose_feature(self, corpus, guess, sentence):
        """
        Debug what's going on
        """

        result = defaultdict(list)
        reverse_vocab = dict((y, x) for x, y in self._vocab.iteritems())

        tokenized = list(self.tokenize_and_censor(sentence))
        norm_title = self.normalize_title(corpus, guess)
        if not norm_title in self._corpora:
            return result
        guess_id = self._corpora[norm_title]

        # Get the counts of words in unigram and bigram
        for ii, jj in bigrams(tokenized):
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
        if not norm_title in self._corpora or self._sentence_length == 0:
            return ""
        else:
            guess_id = self._corpora[norm_title]
            if not guess_id in self._loaded_lms:
                self._lm.read_counts("%s/%i" % (self._datafile, guess_id))
                self._loaded_lms.add(guess_id)

            return self._lm.feature(corpus, guess_id, self._sentence,
                                    self._sentence_length)


class LanguageModelWriter(LanguageModelBase):
    def __init__(self, vocab_size, comparison_corpora):
        self._vocab_size = vocab_size
        self._vocab_final = False
        self._vocab = {}
        self._compare = comparison_corpora

        self._training_counts = DistCounter()
        self._obs_counts = {}

        # Unigram counts
        self._unigram = defaultdict(DistCounter)

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
            self._vocab = dict((x, y + 1) for y, x in enumerate(vocab))

            assert not kUNK in self._vocab, "Vocab already has %s" % kUNK
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

        if not corpus in self._obs_counts:
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
        for ii in xrange(self._compare):
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
            print("Done writing vocab")

        corpus_num = 0
        for cc in self.corpora():
            outfile.write("%s %i\n" % (cc, self.compare(cc)))

            if corpus_num % 100 == 0:
                print(cc, self.compare(cc))

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
                    "Got %s for %s %s" % (self._obs_counts[corpus][ii][jj],
                                          corpus, ii, jj)

                outfile.write("%i %i\n" %
                              (jj, self._obs_counts[corpus][ii][jj]))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--wiki_location', type=str, default='data/wikipedia')
    parser.add_argument('--question_db', type=str, default='data/questions.db')
    parser.add_argument('--source_location', type=str, default='data/source')
    parser.add_argument('--global_lms', type=int, default=5,
                        help="The number of background LMs we maintain")
    parser.add_argument('--vocab_size', type=int, default=100000)
    parser.add_argument("--min_answers", type=int, default=1,
                        help="How many answers needed before including in LM")
    parser.add_argument("--max_pages", type=int, default=-1,
                        help="How many pages to add to the index")
    parser.add_argument("--stats_pages", type=int, default=5000,
                        help="How many pages to use for computing stats")
    parser.add_argument("--lm_out", type=str, default='data/language_model')
    flags = parser.parse_args()

    min_answers = flags.min_answers
    print("Training language model with pages that appear more than %i times" %
          min_answers)

    lm = LanguageModelWriter(flags.vocab_size, flags.global_lms)
    num_docs = 0
    background = defaultdict(int)
    # Initialize language models
    for title, text in text_iterator(True, flags.wiki_location,
                                     True, flags.question_db,
                                     True, flags.source_location,
                                     flags.max_pages,
                                     min_pages=min_answers):
        num_docs += 1
        if num_docs % 500 == 0:
            print(unidecode(title), num_docs)
            print(list(lm.tokenize_without_censor(text[100:200])))

        for tt in lm.tokenize_without_censor(text):
            background[tt] += 1

    # Create the vocabulary
    for ii in background:
        lm.train_seen(ii, background[ii])
    vocab = lm.finalize()
    print(str(vocab)[:80])
    print("Vocab size is %i from %i docs" %
          (len(vocab), num_docs))
    del background

    # Train the language model
    doc_num = 0
    for corpus, qb, wiki, source in [("wiki", False, True, False),
                                     ("qb", True, False, False),
                                     ("source", False, False, True)
                                     ]:
        # Add training data
        start = time.time()
        for title, text in text_iterator(wiki, flags.wiki_location,
                                         qb, flags.question_db,
                                         source, flags.source_location,
                                         flags.max_pages,
                                         min_pages=min_answers):
            doc_num += 1
            if doc_num % 500 == 0 or time.time() - start > 10:
                print("Adding train doc %i, %s (%s)" %
                      (doc_num, unidecode(title), corpus))
                start = time.time()
            lm.add_train(corpus, title, text)

    print("Done training")
    if flags.lm_out:
        # Create the extractor object and write out the pickle
        o = open("%s.txt" % flags.lm_out, 'w')
        lm.write_vocab(o)
        o.close()

        os.mkdir("%s" % flags.lm_out)
        for ii, cc in enumerate(lm.corpora()):
            o = open("%s/%i" % (flags.lm_out, ii), 'w')
            lm.write_corpus(cc, ii, o)
            o.close()
