from qanta import logging
log = logging.get(__name__)


class PageAssigner:
    def __init__(self, normalize_func=lambda x: x):
        from collections import defaultdict, Counter

        self._unambiguous = {}
        self._ambiguous = defaultdict(dict)
        self._direct = defaultdict(dict)

        self._normalize = normalize_func

        self._counts = defaultdict(Counter)

    def load_unambiguous(self, filename):
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

                if answer in self._unambiguous:
                    log.info("%s in unambiguous twice" % answer)
                    assert self._unambiguous[answer] == page, \
                        "%s inconsistent in unambiguous (%s vs %s)" % \
                        (answer, page, self._unambiguous[answer])

                self._unambiguous[answer] = page

    def load_ambiguous(self, filename):
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

                words = words.split(":")
                self._ambiguous[answer][page] = words

    def load_direct(self, filename):
        with open(filename) as infile:
            for ii in infile:
                if not ii.strip():
                    continue
                try:
                    ext_id, page, answer = ii.strip().split('\t')
                except ValueError:
                    log.info("Bad direct line in %s: %s" % (filename, ii))
                self._direct[ext_id] = page

    def known_pages(self):
        pages = set(self._direct.values())
        for ii in self._ambiguous:
            pages |= set(self._ambiguous[ii].keys())
        pages |= set(self._unambiguous.values())
        return pages

    def __call__(self, answer, text, pb="", naqt=-1):
        normalize = self._normalize(answer)

        if pb in self._direct:
            val = self._direct[pb]
            self._counts["D-P"][val] += 1
            return val

        if naqt in self._direct:
            val = self._direct[naqt]
            self._counts["D-N"][val] += 1
            return val

        if normalize in self._unambiguous:
            val = self._unambiguous[normalize]
            self._counts["U"][val] += 1
            return val

        if normalize in self._ambiguous:
            default = [x for x in self._ambiguous[normalize] if
                       len(self._ambiguous[normalize]) == 0]
            assert len(default) <= 1, "%s has more than one default" % normalize
            assert len(default) < len(self._ambiguous[normalize]), "%s only has default" % normalize

            # See if any words match
            words = None
            for jj in self._ambiguous[normalize]:
                for ww in self._ambiguous[normalize][jj]:
                    if words is None:
                        words = set(text)
                    if ww in [x.lower() for x in words]:
                        val = jj
                        self._counts["A"][val] += 1

            # Return default if there is one
            if len(default) == 1:
                val = default[0]
                self._counts["A"][val] += 1
                return val
            else:
                return ''

        # Give up if we can't find answer
        return ''

    def get_counts(self):
        return self._counts
