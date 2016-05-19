
from glob import glob
import urllib2
import sys
from urllib2 import HTTPError
import pickle
import os

from time import sleep

from bs4 import BeautifulSoup
from unidecode import unidecode

if __name__ == "__main__":
    if os.path.isfile("cache.pkl"):
        cache = pickle.load(open("cache.pkl", 'rb'))
        print("Read %i from cache" % len(cache))
        sleep(10)
    else:
        cache = {}

    buffer = ""
    for ii in glob("%s/*" % sys.argv[1]):
        pickle.dump(cache, open("%s.pkl" % sys.argv[1], 'wb'))
        for line, jj in enumerate(open(ii)):
            print(jj)
            if jj in cache:
                buffer += cache[jj]
            else:
                sleep(15)
                try:
                    res = urllib2.urlopen('http://www.poemhunter.com/%s' % jj.strip())
                    html = unidecode(res.read())
                except HTTPError:
                    print("Error!")
                    sleep(60)
                    continue

                try:
                    text = html.split('<div class="KonaBody" style="padding-right:5px">')[1]
                    text = text.split("<!-- .KonaBody -->")[0]

                    if "</script>" in text:
                        text = text.rsplit("</script>", 1)[1]

                    soup = BeautifulSoup(text, 'html.parser')
                    cache[jj] = "%s\n\n\n" % jj.replace("-", " ").replace("poem/", "")

                    print("%s %i %s" % (sys.argv[1], line,
                                        ":".join(x.strip() for x in
                                                 list(soup.strings)[:5])))
                    cache[jj] += " ".join(soup.strings)
                    buffer += "%s\n\n" % cache[jj]
                except IndexError:
                    print("Index error")

    o = open("%s.txt" % sys.argv[1], 'w')
    o.write(unidecode(buffer))
