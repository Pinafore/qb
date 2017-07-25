import os
import csv
import pickle
from multiprocessing import Pool
from itertools import chain
import time
from collections import ChainMap, namedtuple
from urllib import parse

import wikipedia

from qanta import logging
from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.util.constants import COUNTRY_LIST_PATH, WIKI_LOCATION, WIKI_DUMP_REDIRECT_PICKLE, WIKI_PAGE_PATH
from qanta.util.io import safe_path
from qanta.config import conf

log = logging.get(__name__)

COUNTRY_SUB = ["History of ", "Geography of "]

WikipediaPage = namedtuple('WikipediaPage', ['title', 'content', 'links', 'categories', 'wiki_id', 'url'])


def create_wiki_page(title, content, links=None, categories=None, wiki_id=None, url=None) -> WikipediaPage:
    return WikipediaPage(title, content, links, categories, wiki_id, url)


def access_page(title, cached_wiki):
    # accessing the page forces it to fetch from wikipedia if it isn't
    # cached so this is not a no-op
    cached_wiki[title].content
    return None


def web_initialize_file_cache(path, remote_delay=1):
    """
    Initialize the cache by requesting each page with wikipedia package.
    This function iterates over all pages and accessing them in the cache. This forces a
    prefetch of all wiki pages
    """
    db = QuestionDatabase()
    pages = db.questions_with_pages()
    cw = CachedWikipedia(path, remote_delay=remote_delay)
    pool = Pool()

    input_data = [(title, cw) for title in pages.keys()]
    pool.starmap(access_page, input_data)


def normalize_wikipedia_title(title):
    """
    Normalize wikipedia title coming from raw dumps. This removes non-ascii characters by converting them to their
    nearest equivalent and replaces spaces with underscores
    :param title: raw wikipedia title
    :return: normalized title
    """
    return title.replace(' ', '_')


def create_wikipedia_title_pickle(dump_path, output_path):
    from qanta.spark import create_spark_session

    spark = create_spark_session()
    wiki_df = spark.read.json(dump_path)
    raw_titles = wiki_df.select('title').distinct().collect()
    clean_titles = {normalize_wikipedia_title(r.title) for r in raw_titles}
    with open(output_path, 'wb') as f:
        pickle.dump(clean_titles, f)
    spark.stop()


def create_wikipedia_cache(dump_path):
    from qanta.spark import create_spark_session

    spark = create_spark_session()
    db = QuestionDatabase()
    answers = set(db.all_answers().values())
    b_answers = spark.sparkContext.broadcast(answers)
    # Paths used in spark need to be absolute and it needs to exist
    page_path = os.path.abspath(safe_path(WIKI_PAGE_PATH))

    def create_page(row):
        title = normalize_wikipedia_title(row.title)
        filter_answers = b_answers.value
        if title in filter_answers:
            page = WikipediaPage(title, row.text, None, None, row.id, row.url)
            write_page(page, page_path=page_path)

    spark.read.json(dump_path).rdd.foreach(create_page)


def create_wikipedia_redirect_pickle(redirect_csv, output_pickle):
    countries = {}
    with open(COUNTRY_LIST_PATH) as f:
        for line in f:
            k, v = line.split('\t')
            countries[k] = v.strip()

    db = QuestionDatabase()
    pages = set(db.all_answers().values())

    with open(redirect_csv) as redirect_f:
        redirects = {}
        n_total = 0
        n_selected = 0
        for row in csv.reader(redirect_f, quotechar='"', escapechar='\\'):
            n_total += 1
            source = row[0]
            target = row[1]
            if (target not in pages or source in countries or
                    target.startswith('WikiProject') or
                    target.endswith("_topics") or
                    target.endswith("_(overview)")):
                continue
            else:
                redirects[source] = target
                n_selected += 1

        log.info('Filtered {} raw wikipedia redirects to {} matching redirects'.format(n_total, n_selected))

    with open(output_pickle, 'wb') as output_f:
        pickle.dump(redirects, output_f)


def title_to_filepath(title: str, page_path=WIKI_PAGE_PATH):
    """
    Convert the page title to a safe filepath to either read or write from
    """
    filename = parse.quote(title, safe='')
    return os.path.join(page_path, filename)


def write_page(page: WikipediaPage, page_path=WIKI_PAGE_PATH):
    """
    Write a WikipediaPage object to the disk cache.

    """

    # We use the title rather than an explicit key to account for redirects. safe='' so that / is url encoded
    filepath = title_to_filepath(page.title, page_path=page_path)

    with open(filepath, 'wb') as f:
        log.info("Writing file to %s" % filepath)
        pickle.dump(page, f)


class CachedWikipedia:
    def __init__(self, location=WIKI_LOCATION, write_dummy=True,
                 remote_fallback=conf['cached_wikipedia_remote_fallback'], remote_delay=1):
        """
        CachedWikipedia provides a unified way and easy way to access Wikipedia pages. Its design is motivated by:
        1) Getting a wikipedia page should function as a simple python dictionary access
        2) It should be able to use pre-cached wikipedia content, but fallback to using the `wikipedia` package to fetch
        content in realtime
        3) It should support access to pages using non-canonical names by resolving them to canonical names

        The following sections explain how the different levels of caching work as well as how redirects work

        Caching
        There are two levels of caching in CachedWikipedia. When a request is issued by the user for a page the
        in memory dictionary-backed cache is checked first. If the content exists it is returned. If it does not exist
        then CachedWikipedia will check if there is a cache file for that page. If the page is "Albert_Einstein" then
        this would be the file "data/external/wikipedia/pages/Albert_Einstein". If it exists then it is loaded into the
        dictionary-backed cache and returned. If that file does not exist and the variable
        remote_fallback=cached_wikipedia_remote_fallback is True (which is not the default), then a HTTP request to
        wikipedia fetches the page content and stores it in both the file-backed and dictionary-backed cache.

        Redirects
        To support some flexibility in requesting pages that are very close matches we have two sources of redirects.
        The first is based on wikipedia database dumps which is the most reliable, the second is a redirect cache which
        should *not* in general be relied on since it requires that the request was attempted in the first place. On top
        of this we do the very light preprocessing step of replacing whitespace with underscores since the canonical
        page names in the wikipedia database dumps contains an underscore instead of whitespace (a difference from the
        HTTP package which defaults to the opposite)

        remote_delay sets the number of seconds to sleep for in between requests, if set to 0 then there is no delay.
        Please be courteous to Wikipedia and rate limit requests if remote_fallback is set to True (by default it is
        False)
        """
        self.root_path = location
        self.dump_redirect_path = WIKI_DUMP_REDIRECT_PICKLE
        self.cached_redirect_path = os.path.join(self.root_path, 'cached_redirects.pkl')
        self.cache = {}
        self.write_dummy = write_dummy
        self.countries = dict()
        self.cached_wikipedia_remote_fallback = remote_fallback
        # TODO: use an actual rate limiter rather than just a delay
        self.remote_delay = remote_delay
        self._dump_redirects = {}
        self._cached_redirects = {}

        if COUNTRY_LIST_PATH:
            with open(COUNTRY_LIST_PATH) as f:
                for line in f:
                    k, v = line.split('\t')
                    self.countries[k] = v.strip()

        if os.path.exists(self.dump_redirect_path):
            with open(self.dump_redirect_path, 'rb') as f:
                self._dump_redirects = pickle.load(f)
        else:
            raise ValueError(
                'The redirect file (%s) from the dump is missing, run: luigi --module qanta.pipeline.preprocess WikipediaRedirectPickle' % self.dump_redirect_path)

        if os.path.exists(self.cached_redirect_path):
            with open(self.cached_redirect_path, 'rb') as f:
                self._cached_redirects = pickle.load(f)

        # This allows for transparrent access to both the requests
        # cached via wikipedia HTTP requests and those defined via the
        # SQL database dump. ChainMap checks the dictionaries for the
        # existence of the key from left to right so _cached_redirects
        # is checked first then _dump_redirects. If a new key is
        # written to the _redirects dictionary it will mutate the
        # first dictionary which is _cached_redirects. Thus it is
        # sufficient to set new keys on _redirects and resave
        # _cached_redirects upon finding new redirects, allowing for
        # _dump_redirects to correctly remain unchanged since it comes
        # from a luigi job
        self._redirects = ChainMap(self._cached_redirects, self._dump_redirects)

    def _wiki_request_page(self, key: str):
        log.info('Loading {}'.format(key))
        try:
            raw = wikipedia.page(key, preload=True)
            time.sleep(self.remote_delay)
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

    def _load_remote_page(self, key):
        if key in self.countries:
            raw = [self._wiki_request_page(key)]
            for ii in ["%s%s" % (x, self.countries[key]) for x in COUNTRY_SUB]:
                raw.append(self._wiki_request_page(ii))

            log.info("%s is a polity!" % key)
        else:
            raw = [self._wiki_request_page(key)]

        raw = [x for x in raw if x is not None]
        if raw:
            if len(raw) > 1:
                log.info("%i pages for %s" % (len(raw), key))
            page = create_wiki_page(key, "\n".join(x.content for x in raw))
            write_page(page)

            if raw[0].title != key:
                self._redirects[key] = raw[0].title
                log.info("%s redirects to %s" % (key, raw[0].title))
                with open(self.cached_redirect_path, 'wb') as f:
                    pickle.dump(self._cached_redirects, f)
            if len(raw) > 1:
                log.info("%i pages for %s" % (len(raw), key))
            page = create_wiki_page(
                raw[0].title,
                content="\n".join(x.content for x in raw),
                links=set(chain(*[x.links for x in raw])),
                categories=set(chain(*[x.categories for x in raw]))
            )

            write_page(page)
        else:
            log.info("Dummy page for %s" % key)
            page = create_wiki_page(key, '')
            if self.write_dummy:
                write_page(page)
        return page

    def redirect(self, key):
        """
        Check to see if if this is a known redirect.  If so, return that
        redirect location.  Otherwise, return original key.
        """
        return self._redirects.get(key, key)

    def __getitem__(self, key: str):
        # Look to see if this is a redirect page we know about.  If
        # so, follow redirect instead of original key
        title = self.redirect(key)

        if title in self.cache:
            page = self.cache[title]
        else:
            filepath = title_to_filepath(title)
            if os.path.exists(filepath):
                page = self._load_from_file_cache(filepath)
            elif self.cached_wikipedia_remote_fallback:
                page = self._load_remote_page(title)
            else:
                if self.write_dummy:
                    log.info(
                        'Writing dummy page for "{}"->"{}", remote callback disabled and write dummy enabled'.format(
                            key, title))
                    page = create_wiki_page(title, '')
                    write_page(page)
                else:
                    raise KeyError('"{}"->"{}" not found and both write dummy and remote callback are disabled'.format(
                        key, title))
            self.cache[title] = page

        return page


def main():
    cw = CachedWikipedia()
    pages = ["NYC", "New York City", "Ft. Collins", "Japan"]

    for ii in pages:
        print(ii, cw.redirect(ii))
        print(cw[ii].content[:80])


if __name__ == "__main__":
    main()
