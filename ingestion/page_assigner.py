from qanta import logging
from unidecode import unidecode
log = logging.get(__name__)

# Answer lines that are differentiated only based on unicode
# difference from another page.  Hopefully not many of these.
kALLOWED_UNICODE_MISMATCH = set(["Hermès",
                                 "Angels_&_Demons",
                                 "The_Amazing_Adventures_of_Kavalier_&_Clay",
                                 "Saumel_Johnson",
                                 "Gauss'_law",
                                 "A&W_Root_Beer",
                                 "Folate",
                                 "Weird_Al_Yankovic",
                                 "Samuel_Johnson",
                                 '"Master_Harold"...and_the_Boys',
                                 "Youngstown_Sheet_&_Tube_Co._v._Sawyer",
                                 "Cassiopeia_(Queen_of_Ethiopia)"])

class PageAssigner:
    def __init__(self, normalize_func=lambda x: x,
                 limit_set=None):
        from collections import defaultdict, Counter

        self._unambiguous = {}
        self._ambiguous = defaultdict(dict)
        self._direct = defaultdict(dict)
        self._limit = limit_set

        self._normalize = normalize_func

        self._counts = defaultdict(Counter)
        self._all_pages = Counter()

    def load_unambiguous(self, filename):
        validation_error = False
        with open(filename) as infile:
            log.info("Opening %s" % filename)
            for ii in infile:
                if not ii.strip():
                    continue
                try:
                    answer, page = ii.strip().split('\t')
                    page = page.replace(" ", "_")
                except ValueError:
                    log.info("Bad unambiguous line in %s: %s" % (filename, ii))

                assert answer not in self._ambiguous, "%s in ambig and unambig" % answer
                if answer in self._unambiguous:
                    log.info("%s in unambiguous twice" % answer)
                    assert self._unambiguous[answer] == page, \
                        "%s inconsistent in unambiguous (%s vs %s)" % \
                        (answer, page, self._unambiguous[answer])

                self._unambiguous[answer] = page
                if not self.validate_wikipedia(page):
                    validation_error = True
                    log.info("Validation error in %s: %s" % (filename, page))
        assert not validation_error, "Validation error in unambig"

    def load_ambiguous(self, filename):
        validation_error = False
        with open(filename) as infile:
            for ii in infile:
                if not ii.strip():
                    continue
                fields = ii.strip().split('\t')

                if not (len(fields) == 2 or len(fields) == 3):
                    log.info("Bad ambiguous line in %s: %s" % (filename, ii))
                    continue
                if len(fields) == 3:
                    answer, page, words = fields
                else:
                    answer, page = fields
                    words = ""

                words = [x for x in words.split(":") if x != '']
                page = page.replace(" ", "_")
                assert answer not in self._unambiguous, \
                    "%s in ambig and unambig" % answer
                self._ambiguous[answer][page] = words
                if not self.validate_wikipedia(page):
                    validation_error = True
                    log.info("Validation error in %s: %s" % (filename, page))
        assert not validation_error

    def load_direct(self, filename):
        validation_error = False
        with open(filename) as infile:
            for ii in infile:
                if not ii.strip():
                    continue
                try:
                    ext_id, page, answer = ii.strip().split('\t')
                    page = page.replace(" ", "_")
                    self._direct[ext_id] = page
                    if not self.validate_wikipedia(page):
                        validation_error = True
                        log.info("Validation error %s: %s" % (filename, page))
                except ValueError:
                    log.info("Bad direct line in %s: %s" % (filename, ii))
        assert not validation_error, "Direct validation errors!"

    def validate_wikipedia(self, page):
        """
        If this is a unicode page, make sure we don't have the non-unicode
        version already.  If we have a limit set, also make sure it is
        in that set.
        """
        norm = unidecode(page)
        if page == norm:
            val = True
        else: # Doesn't match normalized form
            if norm not in self._all_pages:
                val = True # Seeing it for the first time
            else:
                if page in kALLOWED_UNICODE_MISMATCH:
                    val = True # It's an exception
                else:
                    val = False # This is bad outcome

        if self._limit and page not in self._limit:
            if page in kALLOWED_UNICODE_MISMATCH:
                val = True
            else:
                val = False
        self._all_pages[page] = 0
        return val

    def known_pages(self):
        pages = set(self._direct.values())
        for ii in self._ambiguous:
            pages |= set(self._ambiguous[ii].keys())
        pages |= set(self._unambiguous.values())
        return pages

    def is_ambiguous(self, answer):
        return self._normalize(answer) in self._ambiguous

    def __call__(self, answer, text, pb="", naqt=-1):
        normalize = self._normalize(answer)

        assert isinstance(pb, str)
        assert isinstance(naqt, int)

        if pb in self._direct:
            val = self._direct[pb].replace(" ", "_")
            self._counts["D-P"][val] += 1
            self._all_pages[val] += 1
            return val

        if naqt in self._direct:
            val = self._direct[naqt].replace(" ", "_")
            self._counts["D-N"][val] += 1
            self._all_pages[val] += 1
            return val

        if normalize in self._unambiguous:
            val = self._unambiguous[normalize].replace(" ", "_")
            self._counts["U"][val] += 1
            self._all_pages[val] += 1
            return val

        if normalize in self._ambiguous:
            default = [x for x in self._ambiguous[normalize] if
                       len(self._ambiguous[normalize][x]) == 0]
            assert len(default) <= 1, "%s has more than one default" % normalize
            assert len(default) < len(self._ambiguous[normalize]), "%s only has default" % normalize

            # See if any words match
            words = None
            for jj in self._ambiguous[normalize]:
                for ww in self._ambiguous[normalize][jj]:
                    if words is None:
                        words = set(text)
                    if ww in [x.lower() for x in words]:
                        val = jj.replace(" ", "_")
                        self._counts["A"][val] += 1
                        self._all_pages[val] += 1
                        return val

            print(self._ambiguous[normalize])
            log.info("Match not found, looking for %s default (%i)" % (normalize, len(default)))

            # Return default if there is one
            if len(default) == 1:
                val = default[0].replace(" ", "_")
                self._counts["A"][val] += 1
                self._all_pages[val] += 1
                return val
            else:
                return ''

        # Give up if we can't find answer
        return ''

    def get_counts(self):
        return self._counts


if __name__ == "__main__":
    import argparse
    from glob import glob
    from qanta.datasets.quiz_bowl import QuestionDatabase

    parser = argparse.ArgumentParser(description='Test page assignment')
    parser.add_argument('--direct_path', type=str,
                        default='data/internal/page_assignment/direct/')
    parser.add_argument('--ambiguous_path', type=str,
                        default='data/internal/page_assignment/ambiguous/')
    parser.add_argument('--unambiguous_path', type=str,
                        default='data/internal/page_assignment/unambiguous/')
    flags = parser.parse_args()


    # Load page assignment information
    pa = PageAssigner(QuestionDatabase.normalize_answer)
    for ii in glob("%s/*" % flags.ambiguous_path):
        pa.load_ambiguous(ii)
    for ii in glob("%s/*" % flags.unambiguous_path):
        pa.load_unambiguous(ii)
    for ii in glob("%s/*" % flags.direct_path):
        pa.load_direct(ii)

    for title, words, pp, nn in [("Chicago", "the city of big shoulders".split(), "", -1),
                                 ("Chicago", "the city of deep pizza".split(), "", -1),
                                 ("Alcohol", "chemistry".split(), "5476da9dea23cca90551b95f", -1),
                                 ("Oklahoma", "sooner".split(), "", -1),
                                 ("o henry", "playwright hair".split(), "", -1),
                                 ("The _Iceman Cometh_", "eugene play".split(), "", 4817)]:
        print("-------------------")
        print(title, words, "|%s|" % pa(title, words, pp, nn))
        counts = pa.get_counts()
        for ii in counts:
            print(ii, counts[ii])
