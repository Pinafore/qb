from collections import defaultdict
import time
import re

from unidecode import unidecode

from nltk.tokenize import RegexpTokenizer
from nltk import bigrams

from util.build_whoosh import text_iterator

import clm

kTOKENIZER = RegexpTokenizer('[A-Za-z0-9]+').tokenize

kUNK = "OOV"
kMAX_TEXT_LENGTH = 5000
kGOODCHAR = re.compile(r"[a-zA-Z0-9]*")


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
        if not isinstance(sentence, basestring):
            sentence = ' '.join(list(sentence))
        for ii in kTOKENIZER(unidecode(sentence)):
            yield self.vocab_lookup(ii.lower())

class LanguageModelReader(LanguageModelBase):
    def __init__(self, lm_file, interp=0.8):
        from clm import intArray

        self._datafile = lm_file
        self._lm = clm.JelinekMercerFeature()
        self._lm.set_interpolation(interp)
        self._sentence = intArray(kMAX_TEXT_LENGTH)
        self._sentence_length = 0
        self._sentence_hash = 0
        self._vocab_final = True

    def init(self):
        infile = open(self._datafile)
        num_lms = int(infile.readline())

        vocab_size = int(infile.readline())
        self._vocab = {}
        for ii in xrange(vocab_size):
            self._vocab[infile.readline().strip()] = ii
        print("Done reading %i vocab (Python)" % vocab_size)

        self._corpora = {}
        for ii in xrange(num_lms):
            line = infile.readline()
            corpus, compare = line.split()
            self._corpora[corpus] = ii

        print(self._corpora.keys()[:10])
        self._lm.read_counts(self._datafile)

    def feature_line(self, corpus, guess, sentence):
        if self._sentence_hash != hash(sentence):
            self._sentence_hash = hash(sentence)
            tokenized = list(self.tokenize_and_censor(sentence))
            self._sentence_length = len(tokenized)
            assert self._sentence_length < kMAX_TEXT_LENGTH
            for ii, ww in enumerate(tokenized):
                self._sentence[ii] = ww

        norm_title = self.normalize_title(corpus, guess)
        if not norm_title in self._corpora or self._sentence_length==0:
            return ""
        else:
            guess_id = self._corpora[norm_title]
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
            vocab = sorted(self._training_counts,
                           key=lambda x: self._training_counts[x],
                           reverse=True)[:self._vocab_size]
            self._vocab = dict((x, y + 1) for y, x in enumerate(vocab))
            assert not kUNK in self._vocab, "Vocab already has %s" % kUNK
            self._vocab[kUNK] = 0
        else:
            self._vocab = vocab

        # -------------------------------------------------
        # Add one for the unknown tokens
        del self._training_counts
        return self._vocab

    def add_train(self, corpus, sentence):
        """
        Add the counts associated with a sentence.
        """

        if not corpus in self._obs_counts:
            self._obs_counts[corpus] = defaultdict(DistCounter)

        for context, word in bigrams(self.tokenize_and_censor(sentence)):
            self._obs_counts[corpus][context].inc(word)
            self._unigram[corpus].inc(word)

    def compare(self, title):
        return hash(title) % self._compare

    def write_lm(self, outfile):
        """
        Write the text-based language model to a file
        """

        # TODO(jbg): actually write the correct mean and variance

        outfile.write("%i\n" % len(self._unigram))
        outfile.write("%i\n" % len(self._vocab))
        vocab_size = len(self._vocab)
        for ii, count in sorted(self._vocab.iteritems(),
                                key=lambda key_value: key_value[1]):
            outfile.write("%s\n" % ii)
        print("Done writing vocab")

        corpus_num = 0
        for ii in sorted(self._unigram):
            corpus_num += 1
            if corpus_num % 100 == 0:
                print("Writing LM for %s (%i of %i)" %
                      (unidecode(ii), corpus_num, len(self._unigram)))

            outfile.write("%s %i\n" % (ii, self.compare(ii)))

        for ii in sorted(self._unigram):
            for jj in xrange(vocab_size):
                if jj in self._obs_counts[ii]:
                    total = self._obs_counts[ii][jj].N()
                    contexts = self._obs_counts[ii][jj].B()
                else:
                    total = 0
                    contexts = 0
                outfile.write("%i %i %i\n" % (jj, total, contexts))
                if jj in self._obs_counts[ii]:
                    for kk in self._obs_counts[ii][jj]:
                        outfile.write("%i %i\n" %
                                      (kk, self._obs_counts[ii][jj][kk]))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--wiki_location', type=str, default='data/wikipedia')
    parser.add_argument('--question_db', type=str, default='data/questions.db')
    parser.add_argument('--source_location', type=str, default='data/source')
    parser.add_argument('--global_lms', type=int, default=5,
                        help="The number of background LMs we maintain")
    parser.add_argument('--vocab_size', type=int, default=100000)
    parser.add_argument("--min_answers", type=int, default=-1,
                        help="How many answers needed before including in LM")
    parser.add_argument("--max_pages", type=int, default=-1,
                        help="How many pages to add to the index")
    parser.add_argument("--stats_pages", type=int, default=5000,
                        help="How many pages to use for computing stats")
    parser.add_argument("--lm_out", type=str, default='data/lm.txt')
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
            norm_title = lm.normalize_title(corpus, title)
            doc_num += 1
            if doc_num % 500 == 0 or time.time() - start > 10:
                print("Adding train doc %i, %s (%s)" %
                      (doc_num, unidecode(title), corpus))
                start = time.time()
            lm.add_train(norm_title, text)
            comp = lm.compare(norm_title)
            for ii in xrange(flags.global_lms):
                if comp != ii:
                    lm.add_train("compare_%i" % ii, text)

    print("Done training")
    if flags.lm_out:
        # Create the extractor object and write out the pickle
        o = open(flags.lm_out, 'w')
        lm.write_lm(o)
