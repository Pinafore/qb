import os
import pickle
from time import sleep
from requests import ConnectionError
from requests.exceptions import ReadTimeout

import wikipedia
from wikipedia.exceptions import WikipediaException
from qanta import logging
from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.util.constants import COUNTRY_LIST_PATH
from qanta.util.environment import QB_QUESTION_DB
from functional import seq

log = logging.get(__name__)

COUNTRY_SUB = ["History of ", "Geography of "]


class LinkResult:
    def __init__(self, text_freq=0, link_freq=0, early=0):
        self.text_freq = text_freq
        self.link_freq = link_freq
        self.early = early

    def componentwise_max(self, lr):
        self.text_freq = max(self.text_freq, lr.text_freq)
        self.link_freq = max(self.link_freq, lr.link_freq)
        self.early = max(self.early, lr.early)

    def any(self):
        """
        Did we find anything on any metric
        """
        return any(x > 0.0 for x in
                   [self.text_freq, self.link_freq, self.early])


class WikipediaPage:
    def __init__(self, content="", links=None, categories=None):
        self.content = content
        self.links = links if links is not None else []
        self.categories = categories if categories is not None else []

    def weighted_link(self, other_page):

        # Get the number of times it's mentioned in text
        no_disambiguation = other_page.split("(")[0].strip()
        if len(self.content) > 0:
            text_freq = self.content.count(no_disambiguation)
            text_freq *= len(no_disambiguation) / float(len(self.content))
        else:
            text_freq = 0.0

        # How many total links are there, divide by that number
        if other_page in self.links:
            link_freq = 1.0 / float(len(self.links))
        else:
            link_freq = 0.0

        # How early is it mentioned in the page
        early = self.content.find(no_disambiguation)
        if early > 0:
            early = 1.0 - early / float(len(self.content))
        else:
            0.0

        return LinkResult(text_freq, link_freq, early)


class CachedWikipedia:
    def __init__(self, location, country_list=COUNTRY_LIST_PATH, write_dummy=True):
        """
        @param write_dummy If this is true, it writes an empty pickle if there
        is an error accessing a page in Wikipedia.  This will speed up future
        runs.
        """
        self.path = location
        self.cache = {}
        self.write_dummy = write_dummy
        self.countries = dict()
        if country_list:
            with open(country_list) as f:
                for line in f:
                    k, v = line.split('\t')
                    self.countries[k] = v

    @staticmethod
    def initialize_cache(path):
        """
        This function iterates over all pages and accessing them in the cache. This forces a
        prefetch of all wiki pages
        """
        db = QuestionDatabase(QB_QUESTION_DB)
        pages = db.questions_with_pages()
        cw = CachedWikipedia(path)
        for p in pages:
            cw[p].content

    def load_page(self, key: str):
        log.info("Loading %s" % key)
        try:
            raw = wikipedia.page(key, preload=True)
            log.info(raw.content[:80])
            log.info(str(raw.links)[:80])
            log.info(str(raw.categories)[:80])
        except KeyError:
            log.info("Key error")
            raw = None
        except wikipedia.exceptions.DisambiguationError:
            log.info("Disambig error!")
            raw = None
        except wikipedia.exceptions.PageError:
            log.info("Page error!")
            raw = None
        except ReadTimeout:
            # Wait a while, see if the network comes back
            log.info("Connection error, waiting 1 minutes ...")
            sleep(60)
            log.info("trying again")
            return CachedWikipedia.load_page(key)
        except ConnectionError:
            # Wait a while, see if the network comes back
            log.info("Connection error, waiting 1 minutes ...")
            sleep(60)
            log.info("trying again")
            return CachedWikipedia.load_page(key)
        except ValueError:
            # Wait a while, see if the network comes back
            log.info("Connection error, waiting 1 minutes ...")
            sleep(60)
            log.info("trying again")
            return CachedWikipedia.load_page(key)
        except WikipediaException:
            # Wait a while, see if the network comes back
            log.info("Connection error, waiting 1 minutes ...")
            sleep(60)
            log.info("trying again")
            return CachedWikipedia.load_page(key)
        return raw

    def __getitem__(self, key: str):
        key = key.replace("_", " ")
        if key in self.cache:
            return self.cache[key]

        if "/" in key:
            filename = "%s/%s" % (self.path, key.replace("/", "---"))
        else:
            filename = "%s/%s" % (self.path, key)
        page = None
        if os.path.exists(filename):
            try:
                page = pickle.load(open(filename, 'rb'))
            except pickle.UnpicklingError:
                page = None
            except AttributeError:
                log.info("Error loading %s" % key)
                page = None
            except ImportError:
                log.info("Error importing %s" % key)
                page = None

        if page is None:
            if key in self.countries:
                raw = [self.load_page("%s%s" % (x, self.countries[key])) for x in COUNTRY_SUB]
                raw.append(self.load_page(key))
                log.info("%s is a country!" % key)
            else:
                raw = [self.load_page(key)]

            raw = [x for x in raw if x is not None]
            if raw:
                if len(raw) > 1:
                    log.info("%i pages for %s" % (len(raw), key))
                page = WikipediaPage(
                    "\n".join(x.content for x in raw),
                    seq(raw).map(lambda x: x.links).flatten().list(),
                    seq(raw).map(lambda x: x.categories).flatten().list())

                log.info("Writing file to %s" % filename)
                pickle.dump(page, open(filename, 'wb'),
                            protocol=pickle.HIGHEST_PROTOCOL)
            else:
                log.info("Dummy page for %s" % key)
                page = WikipediaPage()
                if self.write_dummy:
                    pickle.dump(page, open(filename, 'wb'),
                                protocol=pickle.HIGHEST_PROTOCOL)

        self.cache[key] = page
        return page
