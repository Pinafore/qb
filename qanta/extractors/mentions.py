from collections import defaultdict
from string import punctuation
from functools import lru_cache

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn

from qanta.pattern3 import pluralize
from qanta.extractors.abstract import AbstractFeatureExtractor
from qanta.util.constants import KEN_LM
from qanta.util.environment import data_path
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.wikipedia.cached_wikipedia import CachedWikipedia
from qanta import logging
from qanta.preprocess import format_guess

from clm.lm_wrapper import kTOKENIZER, LanguageModelBase


log = logging.get(__name__)


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


def build_lm_data(output):
    cw = CachedWikipedia()

    with open(output, 'w') as o:
        dataset = QuizBowlDataset(1)
        training_data = dataset.training_data()
        train_pages = {format_guess(g) for g in training_data[1]}
        for i, page in enumerate(train_pages):
            content = cw[page].content
            for sentence in nltk.sent_tokenize(content):
                o.write("%s\n" % " ".join(kTOKENIZER(sentence.lower())))

            if i % 1000 == 0:
                log.info("%i\t%s" % (i, page))


def find_references(sentence, padding=5):
    tags = nltk.pos_tag(word_tokenize(sentence))
    tags.append(("END", "V"))
    states = get_states()

    references_found = []
    this_ref_start = -1
    for i, pair in enumerate(tags):
        word, tag = pair
        if word.lower() == 'this' or word.lower() == 'these':
            this_ref_start = i
        elif all(x in punctuation for x in word):
            continue
        elif word in states:
            continue
        elif this_ref_start >= 0 and tag.startswith('NN') and \
                not tags[i + 1][1].startswith('NN'):
            references_found.append((this_ref_start, i))
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


class Mentions(AbstractFeatureExtractor):
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.refex_count = defaultdict(int)
        self.refex_lookup = defaultdict(set)
        self._lm = None
        self.generate_refexs()
        self.kenlm_path = data_path(KEN_LM)

    @property
    def name(self):
        return 'mentions'

    @property
    def lm(self):
        import kenlm
        if not self.initialized:
            self._lm = kenlm.LanguageModel(self.kenlm_path)
            self.initialized = True
        return self._lm

    def score_guesses(self, guesses, text):
        # Find mentions if the text has changed
        pre = []
        ment = []
        suf = []
        for p, m, s in find_references(text):
            # Exclude too short mentions
            if len(m.strip()) > 3:
                pre.append(p.lower())
                suf.append(s.lower())
                ment.append(m.lower())

        for guess in guesses:
            # Find prefixes, suffixes, and mentions
            best_score = float("-inf")
            for ref in self.referring_exs(guess):
                for pp, ss in zip(pre, suf):
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

            norm_title = LanguageModelBase.normalize_title('', guess)
            assert ":" not in norm_title
            for m in ment:
                m = m.replace(':', '')
                res += " "
                res += ("%s~%s" % (norm_title, m)).replace(" ", "_")

            yield res

    def generate_refexs(self):
        """
        Given all of the possible answers, generate the referring expressions to
        store in dictionary.
        """
        answer_list = {ans for ans in QuizBowlDataset(1).training_data()[1]}
        for raw_answer in answer_list:
            page = format_guess(raw_answer)
            ans = raw_answer.split("_(")[0].lower()
            answer_words = ans.split()
            if len(answer_words) > 1:
                for word in answer_words:
                    # each word and plural form of each word
                    self.refex_lookup[page].add(word)
                    self.refex_lookup[page].add(pluralize(word))
                    self.refex_count[word] += 1
                    self.refex_count[pluralize(word)] += 1

            # answer and plural form
            self.refex_count[ans] += 1
            self.refex_count[pluralize(ans)] += 1
            self.refex_lookup[page].add(ans)
            self.refex_lookup[page].add(pluralize(ans))

            # THE answer
            self.refex_count["the %s" % ans] += 1
            self.refex_lookup[page].add("the %s" % ans)

    def referring_exs(self, answer, max_count=5):
        """

        Given a Wikipedia page, generate all of the referring expressions.
        Right now just rule-based, but should be improved.
        """
        for ii in self.refex_lookup[answer]:
            if self.refex_count[ii] < max_count:
                yield ii
