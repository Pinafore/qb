import os
import pickle
from multiprocessing import Pool

import wikipedia

from qanta import logging
from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.util.constants import COUNTRY_LIST_PATH
from qanta.util.environment import QB_QUESTION_DB, QB_WIKI_LOCATION
from qanta.preprocess import format_guess
from qanta.config import conf
from qanta.spark import create_spark_session

log = logging.get(__name__)

COUNTRY_SUB = ["History of ", "Geography of "]


class WikipediaPage:
    def __init__(self, title, content, wiki_id=None, url=None):
        self.title = title
        self.content = content
        self.wiki_id = wiki_id
        self.url = url


def access_page(title, cached_wiki):
    # accessing the page forces it to fetch from wikipedia if it isn't cached so this is not a no-op
    cached_wiki[title].content
    return None


def spark_initialize_file_cache():
    """
    Initialize the cache using spark and the full wikipedia dump
    :return: 
    """
    spark = create_spark_session()
    wiki_path = 's3a://pinafore-us-west-2/public/wikipedia-json/*/*'
    wiki_df = spark.read.json(wiki_path)
    return wiki_df


def web_initialize_file_cache(path):
    """
    Initialize the cache by requesting each page with wikipedia package.
    This function iterates over all pages and accessing them in the cache. This forces a
    prefetch of all wiki pages
    """
    db = QuestionDatabase(QB_QUESTION_DB)
    pages = db.questions_with_pages()
    cw = CachedWikipedia(path)
    pool = Pool()

    input_data = [(format_guess(title), cw) for title in pages.keys()]
    pool.starmap(access_page, input_data)


class CachedWikipedia:
    def __init__(self, location=QB_WIKI_LOCATION, country_list=COUNTRY_LIST_PATH, write_dummy=True):
        """
        @param write_dummy If this is true, it writes an empty pickle if there
        is an error accessing a page in Wikipedia.  This will speed up future
        runs.
        """
        self.path = location
        self.cache = {}
        self.write_dummy = write_dummy
        self.countries = dict()
        self.cached_wikipedia_remote_fallback = conf['cached_wikipedia_remote_fallback']
        if country_list:
            with open(country_list) as f:
                for line in f:
                    k, v = line.split('\t')
                    self.countries[k] = v

    @staticmethod
    def _wiki_request_page(key: str):
        log.info('Loading {}'.format(key))
        try:
            raw = wikipedia.page(key, preload=True)
            log.info(raw.content[:80])
            log.info(str(raw.links)[:80])
            log.info(str(raw.categories)[:80])
        except KeyError:
            log.info('Key error: {}'.format(key))
            raw = None
        except wikipedia.exceptions.DisambiguationError:
            log.info('Disambig error: {}'.format(key))
            raw = None
        except wikipedia.exceptions.PageError:
            log.info('Page error: {}'.format(key))
            raw = None
        return raw

    @staticmethod
    def _load_from_file_cache(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def _load_remote_page(self, key, filename):
        if key in self.countries:
            raw = [self._wiki_request_page("%s%s" % (x, self.countries[key])) for x in COUNTRY_SUB]
            raw.append(self._wiki_request_page(key))
            log.info("%s is a country!" % key)
        else:
            raw = [self._wiki_request_page(key)]

        raw = [x for x in raw if x is not None]
        if raw:
            if len(raw) > 1:
                log.info("%i pages for %s" % (len(raw), key))
            page = WikipediaPage(key, "\n".join(x.content for x in raw))

            log.info("Writing file to %s" % filename)
            with open(filename, 'wb') as f:
                pickle.dump(page, f)
        else:
            log.info("Dummy page for %s" % key)
            page = WikipediaPage(key, '')
            if self.write_dummy:
                with open(filename, 'wb') as f:
                    pickle.dump(page, f)
        return page

    def __getitem__(self, key: str):
        key = format_guess(key)
        if key in self.cache:
            return self.cache[key]

        if "/" in key:
            filename = os.path.join(self.path, key.replace("/", "---"))
        else:
            filename = os.path.join(self.path, key)

        page = None
        if os.path.exists(filename):
            page = self._load_from_file_cache(filename)

        if page is None:
            if self.cached_wikipedia_remote_fallback:
                page = self._load_remote_page(key, filename)
            else:
                log.info('Could not find local page for {} and remote wikipedia fallback is disabled'.format(key))
                page = WikipediaPage(key, '')

        self.cache[key] = page
        return page
