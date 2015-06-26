from collections import defaultdict
import xml.etree.ElementTree as ET
import argparse
from glob import glob
from math import log

from wikipedia.exceptions import DisambiguationError
from feature_extractor import FeatureExtractor
from util.cached_wikipedia import CachedWikipedia
from extractors.lm import good_char


class WikiLinks(FeatureExtractor):
    def __init__(self, xml_location="data/wikifier/data/output",
                 wikipedia="data/wikipedia"):
        self._name = "wikilinks"
        self._location = xml_location
        self._links = defaultdict(dict)
        self._wiki = CachedWikipedia(wikipedia)

        self._cache = -1
        self._matches = None

    def vw_from_title(self, title, text):
        if hash(text) != self._cache:
            self._cache = hash(text)
            self._matches = set()

            for ii in xrange(self._sent):
                self._matches = self._matches | \
                    set(x[0] for x in
                        self._links[self._qnum].get(ii, {}).values())

            for jj in self._links[self._qnum].get(self._sent, []):
                title, pos, index, score = self._links[self._qnum][ii][jj]
                if self._token > pos:
                    self._matches.add(title)

        total = 0
        matches = ["|%s" % self._name]
        display_title = "".join(x for x in good_char.findall(title) if x)
        for ii in [x for x in self._matches]:
            try:
                page = self._wiki[ii]
            except DisambiguationError:
                continue
            if title in page.links:
                norm = "".join(x for x in good_char.findall(ii) if x)
                matches.append("%s_%s" % (norm, display_title))
                total += 1
        matches.append("Total:%f" % log(1 + total))
        return " ".join(matches)

    def load_xml(self, question):
        for ii in glob("%s/%i-*.txt.wikification.tagged.full.xml" %
                        (self._location, question)):
            sentence = int(ii.split(".txt.wikification")[0].rsplit("-", 1)[-1])
            tree = ET.parse(ii)
            root = tree.getroot()

            # text = tree.find("InputText").text
            # print(text)

            for child in root[2].findall('Entity'):
                surface = child.find("EntitySurfaceForm").text
                start = int(child.find("EntityTextStart").text)
                end = int(child.find("EntityTextEnd").text)

                entity = child.find("TopDisambiguation")
                page = entity.find("WikiTitle").text
                id = entity.find("WikiTitleID").text
                score = float(entity.find("RankerScore").text)

                if not sentence in self._links[question]:
                    self._links[question][sentence] = {}
                self._links[question][sentence][surface] = \
                    (page, start, id, score)
        print self._links[question]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--xml_location", type=str,
                        default="data/wikifier/data/output",
                        help="Where we write output file")
    flags = parser.parse_args()

    wl = WikiLinks(flags.xml_location)

    wl.load_xml(111111)
    wl.set_metadata("", "", 111111, 10, 0, 50, "train")
    print wl._qnum
    print wl.vw_from_title("Melampus", "")
