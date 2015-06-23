import os
import cPickle as pickle
import time
from time import sleep
from requests import ConnectionError

from unidecode import unidecode
import wikipedia, fileinput

class CachedWikipedia:
    def __init__(self, location):
        self._path = location
        self._cache = {}

    def __getitem__(self, key):
        if "/" in key:
            key = key.replace("/", "---")
        key = key.replace("_", " ")
        filename = "%s/%s" % (self._path, key)
        page = None
        if os.path.exists(filename):
            try:
                page = pickle.load(open(filename, 'rb'))
            except pickle.UnpicklingError:
                page = None

        if page is None:
            print("Loading %s" % key)
            try:
                page = wikipedia.page(key, preload=True)
                print(unidecode(page.content[:80]))
                print(unidecode(str(page.links)[:80]))
                print(unidecode(str(page.categories)[:80]))
            except KeyError:
                page = wikipedia.page(key)
            except wikipedia.exceptions.DisambiguationError:
                page = wikipedia.page(key.replace(" ", "_"), preload=True)
            except wikipedia.exceptions.PageError:
                page = wikipedia.page(key.replace(" ", "_"), preload=True)
            except ConnectionError:
                # Wait a while, see if the network comes back
                print("Connection error, waiting 10 minutes ...")
                sleep(600)
                print("trying again")
                return self[key]
            sleep(.3)
            print("Writing file to %s" % filename)
            pickle.dump(page, open(filename, 'wb'))

            # Directly modifying pickled files results in an unpickling error,
            # dont' do this!

            # replace prefix "aV" in links for line in
            # fileinput.input(filename, inplace=True): print line.replace("aV",
            # "")
        return page

if __name__ == "__main__":
    cw = CachedWikipedia("data/wikipedia")
    for ii in ["Camille Saint-Saens", "Napoleon", "Langston Hughes", "Whigs_(British_political_party)", "Carthage"]:
        print("~~~~~")
        print(ii)
        start = time.time()
        print cw[ii].content[:80]
        print str(cw[ii].links)[:80]
        print(time.time() - start)
