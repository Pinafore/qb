from glob import glob
from collections import defaultdict
from string import punctuation

import kenlm
import nltk
from unidecode import unidecode
from pattern.en import pluralize
from nltk.tokenize import word_tokenize

from feature_extractor import FeatureExtractor
from util.cached_wikipedia import CachedWikipedia
from clm.lm_wrapper import kTOKENIZER, LanguageModelBase
from util.build_whoosh import text_iterator

from nltk.corpus import wordnet as wn
kSTATES = set()
for ii in wn.synset("American_state.n.1").instance_hyponyms():
    for jj in ii.lemmas():
        if len(jj.name()) > 2 and not "_" in jj.name():
            kSTATES.add(jj.name())
        elif jj.name().startswith("New_"):
            kSTATES.add(jj.name().replace("New_", ""))

kDEMO_SENT = ["A 2011 play about this character was produced in collaboration between Rokia Traore, Peter Sellars, and Toni Morrison.",
              "The founder of this movement was inspired to develop its style by the stained glass windows he made for the De Lange House.",
              "Calvin Bridges sketched a specific type of these structures that contain diffuse regions called Balbiani rings and puffs.",
              "This group is represented by a dove in the Book of the Three Birds, written by a Welsh member of this group named Morgan Llwyd. A member of this religious group adopted the pseudonym 'Martin Marprelate' to pen a series of attacks against authorities.",
              "This leader spent three days in house arrest during an event masterminded by the 'Gang of Eight.'"]
kDEMO_GUESS = ["Desdemona", "De Stijl", "Mikhail Gorbachev", "Chromosome"]


def find_references(sentence, padding=5):
    tags = nltk.pos_tag(word_tokenize(sentence))
    tags.append(("END", "V"))

    references_found = []
    this_ref_start = -1
    for ii, pair in enumerate(tags):
        word, tag = pair
        if word.lower() == 'this' or word.lower() == 'these':
            this_ref_start = ii
        elif all(x in punctuation for x in word):
            continue
        elif word in kSTATES:
            continue
        elif this_ref_start >= 0 and tag.startswith('NN') and \
                not tags[ii + 1][1].startswith('NN'):
            references_found.append((this_ref_start, ii))
            this_ref_start = -1
        elif tag.startswith('V'):
            this_ref_start = -1

    for start, stop in references_found:
        yield (" ".join(LanguageModelBase.normalize_title(x[0])
                        for x in tags[max(0, start - padding):start]),
               " ".join(LanguageModelBase.normalize_title(x[0])
                        for x in tags[start:stop + 1]),
               " ".join(LanguageModelBase.normalize_title(x[0])
                        for x in tags[stop + 1:stop + padding + 1]))


def build_lm_data(path="data/wikipedia", output="temp/wiki_sent"):
    import nltk
    cw = CachedWikipedia(path, "")
    o = open(output, 'w')

    count = 0
    for ii in [x.split("/")[-1] for x in glob("%s/*" % path)]:
        count += 1
        if count % 1000 == 0:
            print("%i\t%s" % (count, unidecode(ii)))
        page = cw[ii]

        for ss in nltk.sent_tokenize(page.content):
            o.write("%s\n" % " ".join(kTOKENIZER(unidecode(ss.lower()))))


class Mentions(FeatureExtractor):
    @staticmethod
    def has_guess():
        return False

    def name(self):
        return self._name

    def __init__(self, db, min_pages, lm="data/kenlm.apra"):
        self._name = "mentions"
        self._refex_count = defaultdict(int)
        self._refex_lookup = defaultdict(set)

        # Get all of the answers
        answers = set(x for x, y in text_iterator(False, "", False, db,
                                                  False, "", limit=-1,
                                                  min_pages=min_pages))
        self.generate_refexs(answers)

        self._text = ""
        self._lm = kenlm.LanguageModel('data/kenlm.arpa')

    def vw_from_title(self, title, text):
        # Find mentions if the text has changed
        if text != self._text:
            self._text = text
            self._pre = []
            self._ment = []
            self._suf = []
            # Find prefixes, suffixes, and mentions
            for pp, mm, ss in find_references(text):
                # Exclude too short mentions
                if len(mm.strip()) > 3:
                    self._pre.append(unidecode(pp.lower()))
                    self._suf.append(unidecode(ss.lower()))
                    self._ment.append(unidecode(mm.lower()))

        best_score = float("-inf")
        for ref in self.referring_exs(title):
            for pp, ss in zip(self._pre, self._suf):
                pre_tokens = kTOKENIZER(pp)
                ref_tokens = kTOKENIZER(ref)
                suf_tokens = kTOKENIZER(ss)

                query_len = len(pre_tokens) + len(ref_tokens) + len(suf_tokens)
                query = " ".join(pre_tokens + ref_tokens + suf_tokens)
                score = self._lm.score(query)
                if score > best_score:
                    best_score = score / float(query_len)
        if best_score > float("-inf"):
            res = "|%s score:%f" % (self._name, best_score)
        else:
            res = "|%s missing:1" % (self._name)

        norm_title = LanguageModelBase.normalize_title(unidecode(title))
        for mm in self._ment:
            res += " "
            res += ("%s~%s" % (norm_title, mm)).replace(" ", "_")

        assert not ":" in res, "%s %s %s" % (title, str(self._ment), res)
        return res

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
                self._refex_lookup[aa].add(jj.lower())
                self._refex_lookup[aa].add(pluralize(jj).lower())
                self._refex_count[jj] += 1
                self._refex_count[pluralize(jj)] += 1

            # answer and plural form
            self._refex_count[ans.lower()] += 1
            self._refex_count[pluralize(ans).lower()] += 1
            self._refex_lookup[aa].add(ans.lower())
            self._refex_lookup[aa].add(pluralize(ans).lower())

            # THE answer
            self._refex_count["the %s" % ans.lower()] += 1
            self._refex_lookup[aa].add("the %s" % ans.lower())

    def referring_exs(self, answer, max_count=5):
        """

        Given a Wikipedia page, generate all of the referring expressions.
        Right now just rule-based, but should be improved.
        """
        for ii in self._refex_lookup[answer]:
            if self._refex_count[ii] < max_count:
                yield ii

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--build_lm_data', default=False, action='store_true',
                        help="Write current subset of wikipedia to build language model")
    parser.add_argument('--demo', default=False, action='store_true',
                        help="Demo mention scoring")
    parser.add_argument('--lm', default='data/kenlm.arpa', type=str,
                        help="Wikipedia language model")
    parser.add_argument("--min_answers", type=int, default=5,
                        help="Min answers")
    parser.add_argument("--db", type=str,
                        default="data/questions.db",
                        help="Location of questions")
    flags = parser.parse_args()

    if flags.build_lm_data:
        build_lm_data()

    if flags.demo:
        ment = Mentions(flags.db, flags.min_answers, flags.lm)

        # Show the mentions
        for ii in kDEMO_GUESS:
            print(ii, list(ment.referring_exs(ii)))

        for ii in kDEMO_SENT:
            print(ii)
            for jj in find_references(ii):
                print("\t%s\t|%s|\t%s" % jj)
            for jj in kDEMO_GUESS:
                print(ment.vw_from_title(jj, ii))
