import os
import urllib
from itertools import chain
from collections import Counter, defaultdict
from qanta.util.constants import STOP_WORDS
from unidecode import unidecode

from fuzzywuzzy import process


def generate_rule(questions_by_page, max_overlap=15):
    """
    Given to lists of counters, find something the appears in all of
    the left examples but none of the right examples (and vice versa).
    This is used to create a decision list to decide ambiguous
    mappings.

    """

    common = {}
    for ii in questions_by_page:
        # Get all the words common in each page
        common[ii] = set(questions_by_page[ii][0].raw_words())
        for jj in questions_by_page[ii][1:]:
            common[ii] = common[ii] & set(jj.raw_words())
    # print("******")
    # print(common)

    for ii in questions_by_page:
        # Abort if we had repeated questions (e.g., too many words
        # were exactly shared)
        if len(common[ii]) > max_overlap:
            return {}
        for jj in questions_by_page:
            if ii != jj:
                common[ii] -= common[jj]
                # print(ii, jj, common[ii])
            common[ii] -= STOP_WORDS

    return common


class TitleFinder:
    def __init__(self, index, normalize=lambda x: x, prune=1500):
        from collections import defaultdict
        import gzip

        self.normalize = normalize
        self._index = defaultdict(set)

        with gzip.open(index) as f:
            line = 0
            for title in f:
                line += 1
                if line == 1:
                    continue

                for word in [normalize(x) for x in
                             title.decode('utf-8').split("_")]:
                    self._index[word].add(title.decode('utf-8').strip())
                if line % 100000 == 0:
                    print(line, title, word, list(self._index[word])[:3])
                    # break

        self._prune = [(x, len(self._index[x]))
                       for x in self._index if len(self._index[x]) > prune]
        print("************************")
        print(self._prune[:50])
        print("************************")
        for ii, jj in self._prune:
            del self._index[ii]

        print("done")

    def query(self, text, num_candidates=2):
        print("~")
        tokens = self.normalize(text).split()
        candidates = set(chain.from_iterable(self._index[x] for x in tokens))
        if tokens[-1].endswith("s"):
            candidates = candidates or self._index[tokens[-1][:-1]]
        print(text)
        candidates = dict((self.normalize(x.replace("_", " ")), x) for x in
                          candidates)
        val = process.extract(text, candidates, limit=num_candidates)
        return val


def best_guess(unassigned, index, min_val=0.5, delta=0.1):
    results = {}
    guess_num = 0
    for ii in unassigned:
        v = index.query(ii, 2)
        if len(v) >= 2:
            top = v[0]
            second = v[1]

            if top[1] - second[1] > delta and top[1] > min_val:
                results[ii] = top[0]
                guess_num += 1
                if guess_num % 1000 == 0:
                    print("Matching", ii, results.get(ii, None))
    return results


def iterate_answers(answers):
    """
    Given a database object,
    """

    ambiguous = {}
    clear = {}
    unassigned = {}

    answer_num = 0
    for normalized in answers:
        pages = Counter()
        for aa, pp in answers[normalized]:
            if pp:
                pages[unidecode(pp)] += 1

        if len(pages) > 1 and min(pages.values()) > 1:
            ambiguous[normalized] = answers[normalized]
        elif len(pages) == 0:
            unassigned[normalized] = answers[normalized]
        elif len(pages) == 1 and pages.most_common()[0][0]:
            clear[normalized] = pages.most_common()[0][0]

        # Not all cases are covered; does not handle ambiguous
        # assignments with only one example

        answer_num += 1
        if answer_num % 10000 == 0:
            print(answer_num, normalized)

    print("Done processing existing answers")
    return ambiguous, unassigned, clear


if __name__ == "__main__":
    import argparse
    from qanta.datasets.quiz_bowl import QuestionDatabase

    parser = argparse.ArgumentParser(description='Create a decision list for qb questions')
    parser.add_argument('--db', default="data/internal/questions.db", type=str)
    parser.add_argument('--wiki_title', default="data/enwiki-latest-all-titles-in-ns0.gz", type=str)
    parser.add_argument('--guess', default="data/internal/guess_page_assignment", type=str)
    parser.add_argument('--unambiguous', default="data/internal/unambiguous_page_assignment", type=str)
    parser.add_argument('--ambiguous', default="data/internal/ambiguous_page_assignment", type=str)
    parser.add_argument('--direct', default="data/internal/direct_assign", type=str)

    args = parser.parse_args()

    qdb = QuestionDatabase(args.db)

    # Download list of all Wikipedia pages if it's not already there
    if not os.path.exists(args.wiki_title):
        urllib.request.urlretrieve("http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-all-titles-in-ns0.gz",
                                   args.wiki_title)

    answers = qdb.normalized_answers()
    index = TitleFinder(args.wiki_title, qdb.normalize_answer)

    a, u, c = iterate_answers(answers)

    print("Processing %i ambiguous" % len(a))
    direct_assign = defaultdict(dict)
    with open(args.ambiguous, 'w') as o:
        for ii in sorted(a):
            questions = defaultdict(list)
            for jj in answers[ii]:
                for kk in qdb.questions_by_answer(jj[0]):
                    if kk.page:
                        questions[unidecode(kk.page)].append(kk)
                        if kk.naqt > 0:
                            direct_assign[kk.page][kk.naqt] = kk.answer
            # print(questions)
            rules = generate_rule(questions)
            # print(rules)

            for jj in rules:
                # print(ii, jj, rules[jj])
                o.write("%s\t%s\t%s\n" % (ii, jj, ":".join(list(rules[jj]))))
                o.flush()
    with open(args.direct, 'w') as o:
        for pp in sorted(direct_assign):
            for nn in direct_assign[pp]:
                o.write("%i\t%s\t%s\n" % (nn, pp, direct_assign[pp][nn]))

    print("Done processing ambiguous")

    with open(args.unambiguous, 'w') as o:
        for ii in sorted(c):
            o.write("%s\t%s\n" % (ii, c[ii]))

    g = best_guess(u, index)
    with open(args.guess, 'w') as o:
        for ii in sorted(g):
            o.write("%s\t%s\n" % (ii, g[ii]))
