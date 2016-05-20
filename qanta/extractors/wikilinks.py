from collections import defaultdict
from xml.etree import ElementTree
import argparse
from glob import glob
from math import log
from numpy import median

from wikipedia.exceptions import PageError
from util.cached_wikipedia import CachedWikipedia, LinkResult
from clm.lm_wrapper import LanguageModelBase
from extractors.abstract import FeatureExtractor
from qanta.extractors.abstract import FeatureExtractor


class WikiLinks(FeatureExtractor):
    def __init__(self,
                 xml_location="data/wikifier/data/output",
                 wikipedia="data/wikipedia",
                 country_list='data/country_list.txt'):
        super(WikiLinks, self).__init__()
        self.name = "wikilinks"
        self._location = xml_location
        self.links = defaultdict(dict)
        self._wiki = CachedWikipedia(wikipedia, country_list)
        self._cache = -1
        self._matches = None

    def set_metadata(self, answer, category, qnum, sent, token, guesses, fold):
        FeatureExtractor.set_metadata\
            (self, answer, category, qnum, sent, token, guesses, fold)
        # print(qnum, sent, token, answer)
        if qnum not in self._links:
            self.load_xml(qnum)

    def vw_from_title(self, title, text):
        if hash(text) != self._cache:
            self._cache = hash(text)
            self._matches = set()

            # Get all the links from previous sentences
            for ii in range(self._sent):
                self._matches = self._matches | \
                    set(x[0] for x in
                        self.links[self._qnum].get(ii, {}).values())

            # Get links from this sentence if they're before the current
            # position
            for jj in self._links[self._qnum].get(self._sent, []):
                title, pos, index, score = \
                    self.links[self._qnum][self._sent][jj]
                if self._token > pos:
                    self._matches.add(title)

        total = 0
        reciprocal = 0
        matches = ["|%s" % self._name]
        norm_title = LanguageModelBase.normalize_title("", title)
        bad = set()

        best = LinkResult()
        results = []

        for ii in self._matches:
            try:
                page = self._wiki[ii]
            except PageError:
                bad.add(ii)
                continue

            link_result = page.weighted_link(ii)
            best.componentwise_max(link_result)

            if link_result.any():
                results.append(link_result)
                matches.append("%s_%s" %
                               (norm_title,
                                LanguageModelBase.normalize_title("", ii)))
                if ii in self._wiki[title].links:
                    reciprocal += 1

                total += 1

        matches.append("Total:%f" % log(1 + total))
        matches.append("BestText:%f" % best.text_freq)
        matches.append("BestLink:%f" % best.link_freq)
        matches.append("BestEarly:%f" % best.early)
        matches.append("Reciprocal:%f" % reciprocal)

        if len(results) > 0:
            matches.append("MedianText:%f" %
                           median(list(x.text_freq for x in results)))
            matches.append("MedianLink:%f" %
                           median(list(x.link_freq for x in results)))
            matches.append("MedianEarly:%f" %
                           median(list(x.early for x in results)))

        return " ".join(matches)

    def load_xml(self, question):
        for ii in glob("%s/%i-*.txt.wikification.tagged.full.xml" %
                        (self._location, question)):
            sentence = int(ii.split(".txt.wikification")[0].rsplit("-", 1)[-1])
            tree = ElementTree.parse(ii)
            root = tree.getroot()

            # text = tree.find("InputText").text
            # print(text)

            for child in root[2].findall('Entity'):
                surface = child.find("EntitySurfaceForm").text
                start = int(child.find("EntityTextStart").text)

                entity = child.find("TopDisambiguation")
                page = entity.find("WikiTitle").text
                id = entity.find("WikiTitleID").text
                score = float(entity.find("RankerScore").text)

                if sentence not in self.links[question]:
                    self.links[question][sentence] = {}
                self.links[question][sentence][surface] = \
                    (page, start, id, score)

    def vw_from_score(self, results):
        pass

