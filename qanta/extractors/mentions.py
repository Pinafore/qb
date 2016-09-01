from glob import glob
from collections import defaultdict
from string import punctuation
from functools import lru_cache

import kenlm
import nltk
from unidecode import unidecode
from qanta.pattern3 import pluralize
from nltk.tokenize import word_tokenize

from qanta.extractors.abstract import FeatureExtractor
from qanta.util.environment import data_path
from qanta.util.constants import KEN_LM
from qanta.wikipedia.cached_wikipedia import CachedWikipedia
from clm.lm_wrapper import kTOKENIZER, LanguageModelBase

from nltk.corpus import wordnet as wn


@lru_cache(maxsize=None)
def get_states():
    states = set()
    for ii in wn.synset("American_state.n.1").instance_hyponyms():
        for jj in ii.lemmas():
            name = jj.name()
            if len(name) > 2 and "_" not in name:
                states.add(name)
            elif name.startswith("New_"):
                states.add(name.replace("New_", ""))
    return states


def find_references(sentence, padding=5):
    tags = nltk.pos_tag(word_tokenize(sentence))
    tags.append(("END", "V"))
    states = get_states()

    references_found = []
    this_ref_start = -1
    for ii, pair in enumerate(tags):
        word, tag = pair
        if word.lower() == 'this' or word.lower() == 'these':
            this_ref_start = ii
        elif all(x in punctuation for x in word):
            continue
        elif word in states:
            continue
        elif this_ref_start >= 0 and tag.startswith('NN') and \
                not tags[ii + 1][1].startswith('NN'):
            references_found.append((this_ref_start, ii))
            this_ref_start = -1
        elif tag.startswith('V'):
            this_ref_start = -1

    for start, stop in references_found:
        yield (" ".join(LanguageModelBase.normalize_title('', x[0])
                        for x in tags[max(0, start - padding):start]),
               " ".join(LanguageModelBase.normalize_title('', x[0])
                        for x in tags[start:stop + 1]),
               " ".join(LanguageModelBase.normalize_title('', x[0])
                        for x in tags[stop + 1:stop + padding + 1]))


def build_lm_data(path, output):
    cw = CachedWikipedia(path, "")
    o = open(output, 'w')

    count = 0
    for i in [x.split("/")[-1] for x in glob("%s/*" % path)]:
        count += 1
        if count % 1000 == 0:
            print("%i\t%s" % (count, unidecode(i)))
        page = cw[i]

        for ss in nltk.sent_tokenize(page.content):
            o.write("%s\n" % " ".join(kTOKENIZER(unidecode(ss.lower()))))


class Mentions(FeatureExtractor):
    def __init__(self, answers):
        super().__init__()
        self.name = "mentions"
        self.answers = answers
        self.initialized = False
        self.refex_count = defaultdict(int)
        self.refex_lookup = defaultdict(set)
        self._lm = None
        self.generate_refexs(self.answers)
        self.pre = []
        self.ment = []
        self.suf = []
        self.text = ""
        self.kenlm_path = data_path(KEN_LM)

    @property
    def lm(self):
        if not self.initialized:
            self._lm = kenlm.LanguageModel(self.kenlm_path)
            self.initialized = True
        return self._lm

    def score_guesses(self, guesses, text):
        # Find mentions if the text has changed
        for guess in guesses:
            if text != self.text:
                self.text = text
                self.pre = []
                self.ment = []
                self.suf = []
                # Find prefixes, suffixes, and mentions
                for pp, mm, ss in find_references(text):
                    # Exclude too short mentions
                    if len(mm.strip()) > 3:
                        self.pre.append(unidecode(pp.lower()))
                        self.suf.append(unidecode(ss.lower()))
                        self.ment.append(unidecode(mm.lower()))

            best_score = float("-inf")
            for ref in self.referring_exs(guess):
                for pp, ss in zip(self.pre, self.suf):
                    pre_tokens = kTOKENIZER(pp)
                    ref_tokens = kTOKENIZER(ref)
                    suf_tokens = kTOKENIZER(ss)

                    query_len = len(pre_tokens) + len(ref_tokens) + len(suf_tokens)
                    query = " ".join(pre_tokens + ref_tokens + suf_tokens)
                    score = self.lm.score(query)
                    if score > best_score:
                        best_score = score / float(query_len)
            if best_score > float("-inf"):
                res = "|%s score:%f" % (self.name, best_score)
            else:
                res = "|%s missing:1" % self.name

            norm_title = LanguageModelBase.normalize_title('', unidecode(guess))
            assert ":" not in norm_title
            for mm in self.ment:
                assert ":" not in mm
                res += " "
                res += ("%s~%s" % (norm_title, mm)).replace(" ", "_")

            yield res, guess

    def generate_refexs(self, answer_list):
        """
        Given all of the possible answers, generate the referring expressions to
        store in dictionary.
        """

        # TODO: Make referring expression data-driven

        for aa in answer_list:
            ans = aa.split("_(")[0]
            for jj in ans.split():
                # each word and plural form of each word
                self.refex_lookup[aa].add(jj.lower())
                self.refex_lookup[aa].add(pluralize(jj).lower())
                self.refex_count[jj] += 1
                self.refex_count[pluralize(jj)] += 1

            # answer and plural form
            self.refex_count[ans.lower()] += 1
            self.refex_count[pluralize(ans).lower()] += 1
            self.refex_lookup[aa].add(ans.lower())
            self.refex_lookup[aa].add(pluralize(ans).lower())

            # THE answer
            self.refex_count["the %s" % ans.lower()] += 1
            self.refex_lookup[aa].add("the %s" % ans.lower())

    def referring_exs(self, answer, max_count=5):
        """

        Given a Wikipedia page, generate all of the referring expressions.
        Right now just rule-based, but should be improved.
        """
        for ii in self.refex_lookup[answer]:
            if self.refex_count[ii] < max_count:
                yield ii
