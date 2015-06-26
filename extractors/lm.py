import argparse
from collections import defaultdict
from string import lower
from math import log, isnan
import time
import re
try:
   import cPickle as pickle
except:
   import pickle

from nltk.tokenize import TreebankWordTokenizer
from nltk import bigrams

from unidecode import unidecode

from numpy import mean, var

from util.build_whoosh import text_iterator
from util.qdb import QuestionDatabase
from extractors.ir import stopwords
from feature_extractor import FeatureExtractor

#kINTERP_CONSTANTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
kINTERP_CONSTANTS = [0.9]
kNEG_INF = -1e6
kTOKENIZER = TreebankWordTokenizer().tokenize

good_char = re.compile(r"[a-zA-Z0-9]*")

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


def display_ngram(chain):
    val = unidecode("_".join(x if x else "OOV" for x in chain))
    val = val.replace(":", "~COLON~")
    val = val.replace("|", "~PIPE~")
    return val


def trim_and_split(chain, stopwords, max_length):
    # Remove stopwords at the start and end, also remove OOV
    content_indices = [ii for ii, ww in enumerate(chain)
                       if not ww in stopwords and ww]
    start = min(content_indices + [len(chain)])
    stop = max([0] + content_indices)
    chain = chain[start:(stop + 1)]
    # print("\t%i %i %s" % (start, stop, str(chain)))
    if all(x in stopwords for x in chain):
        return

    # Return multiple chains if it's too long
    if len(chain) > max_length:
        for start in xrange(len(chain) - max_length + 1):
            yield chain[start:(start + max_length)]
    elif len(chain) > 1:
        yield chain


class LanguageModel(FeatureExtractor):
    def __init__(self, global_lms, threshold=-1):
        self._lm = {}
        self._sent_mean = {}
        self._sent_var = {}
        self._ngram_mean = defaultdict(dict)
        self._ngram_var = defaultdict(dict)
        self._name = "lm"
        self._globals = global_lms
        self._last_sent = ""
        self._cache = 0
        self._threshold = threshold

    def add_corpus(self, corpus_name, lm, qb_location, max_pages):
        self._lm[corpus_name] = lm
        self._set_stats(corpus_name, lm, qb_location, max_pages)

    def _set_stats(self, corpus, lm, qb_location, max_pages):
        sents = []
        ngrams = defaultdict(list)

        qdb = QuestionDatabase(qb_location)
        pages = qdb.questions_with_pages()

        print("Computing stats for %s from %i pages ..." % (corpus, max_pages))
        page_count = 0
        for pp in sorted(pages, key=lambda k: len(pages[k]),
                         reverse=True):
            compare = (hash(pp) + 1) % self._globals
            page_count += 1
            for qq in [x for x in pages[pp] if x.fold == "dev"]:
                if max_pages > 0 and page_count > max_pages:
                    break
                if page_count % 34 == 0:
                    print("%i\t%s" % (page_count, pp))
                for ss in qq.text_lines():
                    if pp in lm:
                        text = list(lm[pp].tokenize_and_censor(ss["text"]))
                        sents.append(lm[pp].mean_ll(text) -
                                     lm[compare].mean_ll(text))

                        for cc in lm[pp].ngram_chains(text):
                            ngrams[len(cc)].\
                                append(lm[pp].mean_ll(cc) -
                                       lm[compare].mean_ll(cc))
        print("done")

        print("Sents", sents[:10])
        self._sent_mean[corpus] = mean(sents)
        self._sent_var[corpus] = var(sents)

        print("Ngrams", ngrams[2][:10])
        for ii in ngrams:
            self._ngram_mean[corpus][ii] = mean(list(x for x in ngrams[ii] if
                                                     x > self._threshold))
            self._ngram_var[corpus][ii] = var(list(x for x in ngrams[ii] if
                                                   x > self._threshold))

        print("Stats for %s: SM: %f, SV: %f, NM: %f, NV: %f" %
              (corpus, self._sent_mean[corpus], self._sent_var[corpus],
               self._ngram_mean[corpus][2], self._ngram_var[corpus][2]))

    #@profile
    def text_score(self, corpus, title, text):
        assert isinstance(text, list)
        sent = self._lm[corpus][title].mean_ll(text)
        compare = (hash(title) + 1) % self._globals
        background = self._lm[corpus][compare].mean_ll(text)
        val = sent - background
        score = (val - self._sent_mean[corpus]) / self._sent_var[corpus]
        return score

    #@profile
    def ngram_score(self, corpus, title, ngram, debug=False):
        if debug:
            print("Computing ngram score for %s in %s (%s)" %
                  (str(ngram), title, corpus))
        # Compute global score
        compare = (hash(title) + 1) % self._globals
        background = self._lm[corpus][compare].mean_ll(ngram)
        # Compute local score
        local = self._lm[corpus][title].mean_ll(ngram)

        score = local - background
        if score > self._threshold:
            ngram_score = (score - self._ngram_mean[corpus][len(ngram)]) \
                / self._ngram_var[corpus][len(ngram)]
        else:
            ngram_score = None

        if debug:
            print("Comparing against %i gave value %f, local=%f" %
                  (compare, background, local))
            print("Score: %f - %f = %f" % (local, background, score))

        return ngram_score

    def print_stats(self):
        bins = 0
        for cc in self._lm:
            for mm in self._lm[cc]:
                for ii in self._lm[cc][mm]._obs_counts:
                    bins += self._lm[cc][mm]._obs_counts[ii].B()
        print("Total bins %i" % bins)

    def set_sentence(self, text):
        """
        Cache tokenized version of sentence if we need to
        """
        if self._cache != hash(text):
            self._cache = hash(text)
            del self._last_sent
            self._last_sent = {}
            for cc in self._lm:
                self._last_sent[cc] = list(self._lm[cc][0].
                                           tokenize_and_censor(text))
        return self._last_sent

    def vw_from_title(self, title, text):
        self.set_sentence(text)
        val = ["|%s" % self._name]
        for corpus in self._lm:
            max_ngram = 0.0
            ngram_count = 0
            if title in self._lm[corpus]:
                norm = "".join(x for x in good_char.findall(title) if x)
                score = self.text_score(corpus, title, self._last_sent[corpus])
                val.append("%s_full:%f" % (corpus, score))

                # We don't want to cheat on training data, so we need a higher
                # threshold for qb data
                if self._fold == "train" and corpus=="qb":
                    cutoff = 1
                else:
                    cutoff = 0

                for cc in self._lm[corpus][title].\
                        ngram_chains(self._last_sent[corpus],
                                     freq_cutoff=cutoff):
                    ngram_score = self.ngram_score(corpus, title, cc)
                    if ngram_score is None:
                        continue
                    max_ngram = max(ngram_score, max_ngram)
                    ngram_count += 1
                    val.append("ngrm_%s_%s_%s" %
                               (corpus, norm, display_ngram(cc)))
            val.append("%s_max:%f" % (corpus, max_ngram))
            val.append("%s_count:%f" % (corpus, log(1 + ngram_count)))
        val = " ".join(val)
        return val

    def verbose(self, qb_location):
        qdb = QuestionDatabase(qb_location)
        pages = qdb.questions_with_pages()
        import time

        for pp in sorted(pages, key=lambda k: len(pages[k]),
                         reverse=True):
            need_title = True
            compare = (hash(pp) + 1) % self._globals
            for corpus in self._lm:
                if not pp in self._lm[corpus]:
                    continue

                for qq in [x for x in pages[pp] if x.fold == "dev"]:
                    if need_title:
                        print("--------------\t%s\t--------------" % pp)
                        need_title = False
                    for ss in qq.text_lines():
                        self.set_metadata(qq.page, qq.category, qq.qnum,
                                          ss["sent"], 0, None, qq.fold)
                        start = time.time()
                        print("===============\t%s\t===============" % corpus)
                        print(self.vw_from_title(pp, ss["text"]))
                        text = list(self._lm[corpus][0].
                                    tokenize_and_censor(ss["text"]))
                        sent = self._lm[corpus][pp].mean_ll(text)
                        background = \
                            self._lm[corpus][compare].mean_ll(text)
                        score = self.text_score(corpus, pp, text)
                        print("sent: ([%f - %f] - %f) / %f = %f" %
                              (sent, background, self._sent_mean[corpus],
                               self._sent_var[corpus], score))

                        for cc in self._lm[corpus][pp].\
                                ngram_chains(text):
                            ngram_score = self.ngram_score(corpus, pp, cc)
                            vv = self._lm[corpus][pp].mean_ll(cc)
                            background = \
                                self._lm[corpus][compare].mean_ll(cc)
                            print("ngram, %s: ([%f - %f] - %f) / %f = %f" %
                                  (display_ngram(cc), vv, background,
                                   self._ngram_mean[corpus][len(cc)],
                                   self._ngram_var[corpus][len(cc)],
                                   ngram_score))
                            print(list(x if x in self._lm[corpus][compare]
                                       ._vocab else None for x in cc))
                        print("TIME: %f" % (time.time() - start))


class JelinekMercerLanguageModel:
    def __init__(self, vocab_size, unigram_smooth=0.01,
                 jm_lambda=0.6, normalize_function=lower):
        self._vocab_size = vocab_size
        self._jm_lambda = jm_lambda
        self._vocab_final = False
        self._vocab = set()
        self._smooth = unigram_smooth

        # Add your code here!
        # Bigram counts
        self._training_counts = DistCounter()
        self._obs_counts = defaultdict(DistCounter)

        # Unigram counts
        self._unigram = DistCounter()

    def train_seen(self, word, count=1):
        """
        Tells the language model that a word has been seen @count times.  This
        will be used to build the final vocabulary.
        """
        assert not self._vocab_final, \
            "Trying to add new words to finalized vocab"

        self._training_counts.inc(word, count)

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
            return word
        else:
            return None

    def finalize(self, vocab=None):
        """
        Fixes the vocabulary as static, prevents keeping additional vocab from
        being added
        """
        self._vocab_final = True
        if vocab is None:
            self._vocab_size = min(len(self._training_counts), self._vocab_size)
            self._vocab = sorted(self._training_counts,
                                 key=lambda x: self._training_counts[x],
                                 reverse=True)[:self._vocab_size]
        else:
            self._vocab = vocab

        # -------------------------------------------------
        # Add one for the unknown tokens
        self._vocab_size = len(self._vocab) + 1
        del self._training_counts
        self._smooth_sum = self._vocab_size * self._smooth
        return self._vocab


    def tokenize_and_censor(self, sentence):
        """
        Given a sentence, yields a sentence suitable for training or testing.
        Prefix the sentence with <s>, replace words not in the vocabulary with
        <UNK>, and end the sentence with </s>.
        """
        if not isinstance(sentence, basestring):
            sentence = ' '.join(list(sentence))
        for ii in kTOKENIZER(sentence):
            yield self.vocab_lookup(ii.lower())

    def tokenize_without_censor(self, sentence):
        for ii in kTOKENIZER(sentence):
            yield ii.lower()

    #@profile
    def ngram_chains(self, sentence, max_length=3, freq_cutoff=0):
        assert isinstance(sentence, list)
        chain = []
        for context, word in bigrams(sentence):
            if context in self._obs_counts and \
                    self._obs_counts[context][word] > freq_cutoff and \
                    (not (context is None) or not (word is None)):
                # print("+%s %s" % (word, str(chain)))
                if not chain:
                    chain.append(context)
                chain.append(word)
            else:
                # print("-%s %s" % (word, str(chain)))
                if chain:
                    for jj in trim_and_split(chain, stopwords, max_length):
                        yield jj
                chain = []

    def set_jm_interp(self, val):
        self._jm_lambda = val

    #@profile
    def jelinek_mercer(self, context, word, debug=False):
        """
        Return the Jelinek-Mercer log probability estimate of a word
        given a context; interpolates context probability with the
        overall corpus probability.
        """
        if context in self._obs_counts:
            bigram = self._obs_counts[context].freq(word)
        else:
            bigram = 0.0
        unigram = (self._unigram[word] + self._smooth) /\
            self._unigram_normalizer
        if debug:
            print("p(%s|%s) = %f; p(%s) = %f" %
                  (word, context, bigram, word, unigram))
        val = self._jm_lambda * bigram + \
            (1 - self._jm_lambda) * unigram
        if val == 0.0:
            return kNEG_INF
        else:
            return log(val)

    def vocab_size(self):
        return len(self._vocab)

    def add_train(self, sentence):
        """
        Add the counts associated with a sentence.
        """

        # You'll need to complete this function, but here's a line of code that
        # will hopefully get you started.
        for context, word in bigrams(self.tokenize_and_censor(sentence)):
            self._obs_counts[context].inc(word)
            self._unigram.inc(word)
        self._unigram_normalizer = self._smooth_sum + self._unigram.N()

    #@profile
    def mean_ll(self, tokens):
        assert isinstance(tokens, list)

        if len(tokens) >= 2:
            val = log((self._unigram[tokens[0]] + self._smooth) /
                      self._unigram_normalizer)
            val += sum(self.jelinek_mercer(context, word)
                       for context, word in bigrams(tokens))
            val /= float(len(tokens))
        else:
            val = kNEG_INF
        if isnan(val):
            return kNEG_INF
        else:
            return val


def choose_jm(lm, params, qb_location, num_globals):
    qdb = QuestionDatabase(qb_location)

    pages = qdb.questions_with_pages()
    scores = defaultdict(float)
    for ll in params:
        for pp in sorted(pages, key=lambda k: len(pages[k]),
                         reverse=True):
            compare = (hash(pp) + 1) % num_globals
            for qq in [x for x in pages[pp] if x.fold == "dev"]:
                for ss in qq.text_lines():
                    lm[compare].set_jm_interp(ll)
                    text = list(lm[compare].tokenize_and_censor(ss["text"]))
                    try:
                        val = lm[compare].ll(text)
                    except OverflowError:
                        val = float("nan")
                    if isnan(val):
                        continue
                    else:
                        scores[ll] += val

    print(scores, max(scores.values()))
    print(scores)

    return [x for x in scores if scores[x] == max(scores.values())][0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--wiki_location', type=str, default='data/wikipedia')
    parser.add_argument('--question_db', type=str, default='data/questions.db')
    parser.add_argument('--global_lms', type=int, default=5,
                        help="The number of background LMs we maintain")
    parser.add_argument('--vocab_size', type=int, default=40000)
    parser.add_argument("--min_answers", type=int, default=-1,
                        help="How many answers needed before including in LM")
    parser.add_argument("--max_pages", type=int, default=-1,
                        help="How many pages to add to the index")
    parser.add_argument("--stats_pages", type=int, default=5000,
                        help="How many pages to use for computing stats")
    parser.add_argument("--lm_out", type=str, default='data/lm.pkl')
    flags = parser.parse_args()

    combined = LanguageModel(flags.global_lms)

    min_answers = flags.min_answers
    print("Training language model with pages that appear more than %i times" %
          min_answers)

    # Remove QB as part of the training to prevent overfitting in VW
    #
    # TODO: make it so that question counts are removed in generating features
    # on train data
    for corpus, qb, wiki in [("wiki", False, True),
                             ("qb", True, False),
                             ]:
        num_docs = 0
        lm = {}
        background = defaultdict(int)
        # Build the vocabulary
        for title, text in text_iterator(wiki, flags.wiki_location,
                                         qb, flags.question_db,
                                         flags.max_pages,
                                         min_pages=min_answers):
            num_docs += 1
            if not title in lm:
                lm[title] = \
                    JelinekMercerLanguageModel(flags.vocab_size,
                                               normalize_function=
                                               lambda x: unidecode(x.lower()))

            for tt in lm[title].tokenize_without_censor(text):
                background[tt] += 1

        for ii in xrange(flags.global_lms):
            lm[ii] =  \
                JelinekMercerLanguageModel(flags.vocab_size,
                                           normalize_function=lambda x:
                                           unidecode(x.lower()))

        # Create the vocabulary
        vocab = None
        for mm in lm:
            if vocab is None:
                # We must explicitly determine vocab once
                for ww in background:
                    lm[mm].train_seen(ww, background[ww])
                vocab = lm[mm].finalize()
            else:
                # thereafter, use the cached version
                lm[mm].finalize(vocab)
        print("Vocab size for %s is %i from %i docs" %
              (corpus, len(vocab), num_docs))

        del background

        # Add training data
        doc_num = 0
        start = time.time()
        for title, text in text_iterator(wiki, flags.wiki_location,
                                         qb, flags.question_db,
                                         flags.max_pages,
                                         min_pages=min_answers):
            doc_num += 1
            if doc_num % 500 == 0 or time.time() - start > 10:
                print("Adding train doc %i, %s" % (doc_num, unidecode(title)))
                start = time.time()
            lm[title].add_train(text)
            lm[hash(title) % flags.global_lms].add_train(text)

        # Select JM parameter if we need to
        if len(kINTERP_CONSTANTS) > 1:
            interp = choose_jm(lm, kINTERP_CONSTANTS, flags.question_db,
                               flags.global_lms)
        else:
            interp = kINTERP_CONSTANTS[0]
        print("Interpolation parameter for %s = %f" % (corpus, interp))

        # Set all of the parameters
        for pp in lm:
            lm[pp].set_jm_interp(interp)

        # Determine the number of pages we look at to compute statistics; if
        # it's less than the number of pages we compute ngrams for, we need to
        # restrict similarly
        if flags.max_pages > 1 and flags.stats_pages > 1:
            max_stats_pages = min(flags.max_pages, flags.stats_pages)
        elif flags.max_pages > 1:
            max_stats_pages = flags.max_pages
        else:
            max_stats_pages = flags.stats_pages
        combined.add_corpus(corpus, lm, flags.question_db,
                            max_stats_pages)

    for ii in xrange(flags.global_lms):
        print("--------------------------------------")
        print("LM %i" % ii)
        print("Vocab: ", list(lm[ii]._vocab)[:10])
        print("Counts: ", lm[ii]._unigram.B())
        print("--------------------------------------")

    # Code for debugging
    # combined.ngram_score("qb", "Nathaniel Hawthorne", ['blithedale', 'romance'], debug=True)
    # combined.ngram_score("qb", "Thomas Hardy", ['mayor', 'of', 'casterbridge'], debug=True)
    # combined.verbose(flags.question_db)

    if flags.lm_out:
        # Create the extractor object and write out the pickle
        o = open(flags.lm_out, 'wb')
        pickle.dump(combined, o, protocol=pickle.HIGHEST_PROTOCOL)
