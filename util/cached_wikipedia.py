import os
import cPickle as pickle
import time
from time import sleep
from requests import ConnectionError
from requests.exceptions import ReadTimeout
from wikipedia.exceptions import WikipediaException

from unidecode import unidecode
import wikipedia, fileinput


class WikipediaPage:
    def __init__(self, content="", links=[], categories=[]):
        self.content = content
        self.links = links
        self.categories = categories


class CachedWikipedia:
    def __init__(self, location, write_dummy=True):
        """
        @param write_dummy If this is true, it writes an empty pickle if there
        is an error accessing a page in Wikipedia.  This will speed up future
        runs.
        """
        self._path = location
        self._cache = {}
        self._write_dummy = write_dummy

    def __getitem__(self, key):
        key = key.replace("_", " ")
        if key in self._cache:
            return self._cache[key]

        if "/" in key:
            filename = "%s/%s" % (self._path, key.replace("/", "---"))
        else:
            filename = "%s/%s" % (self._path, key)
        page = None
        if os.path.exists(filename):
            try:
                page = pickle.load(open(filename, 'rb'))
            except pickle.UnpicklingError:
                page = None
            except AttributeError:
                print("Error loading %s" % key)
                page = None

        if page is None:
            print("Loading %s" % key)
            try:
                raw = wikipedia.page(key, preload=True)
                print(unidecode(raw.content[:80]))
                print(unidecode(str(raw.links)[:80]))
                print(unidecode(str(raw.categories)[:80]))
            except KeyError:
                print("Key error")
                raw = None
            except wikipedia.exceptions.DisambiguationError:
                print("Disambig error!")
                raw = None
            except wikipedia.exceptions.PageError:
                print("Page error!")
                raw = None
            except ReadTimeout:
                # Wait a while, see if the network comes back
                print("Connection error, waiting 10 minutes ...")
                sleep(600)
                print("trying again")
                return self[key]
            except ConnectionError:
                # Wait a while, see if the network comes back
                print("Connection error, waiting 10 minutes ...")
                sleep(600)
                print("trying again")
                return self[key]
            except ValueError:
                # Wait a while, see if the network comes back
                print("Connection error, waiting 10 minutes ...")
                sleep(600)
                print("trying again")
                return self[key]
            except WikipediaException:
                # Wait a while, see if the network comes back
                print("Connection error, waiting 10 minutes ...")
                sleep(600)
                print("trying again")
                return self[key]


            sleep(.3)
            if raw:
                page = WikipediaPage(unidecode(raw.content),
                                     list(raw.links),
                                     list(raw.categories))
                print("Writing file to %s" % filename)
                pickle.dump(page, open(filename, 'wb'))
            else:
                print("Dummy page for %s" % key)
                page = WikipediaPage()
                if self._write_dummy:
                    pickle.dump(page, open(filename, 'wb'))

        self._cache[key] = page
        return page

if __name__ == "__main__":
    cw = CachedWikipedia("data/wikipedia")
    for ii in ["Camille_Saint-Saens", "Napoleon", "Langston Hughes", "Whigs_(British_political_party)", "Carthage", "Stanwix"]:
        print("~~~~~")
        print(ii)
        start = time.time()
        print cw[ii].content[:80]
        print str(cw[ii].links)[:80]
        print(time.time() - start)
