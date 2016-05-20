# -*- coding: utf-8 -*-

# Given a pickled index of wikipedia pages, provides full-text access.  Also
# allows you to find the "closest" wikipedia page given a title and text.
import gzip
import codecs
import argparse
from glob import glob
from collections import defaultdict

from unidecode import unidecode

from qanta.util.build_whoosh import text_iterator

kWORK_TYPES = ["poem", "novel", "play", "film"]
kENCYCLOPEDIAS = [u"1911_Encyclopædia_Britannica",
                  u"The New Student's Reference Work",
                  u"The_Encyclopedia_Americana_(1906)"]

kEXCLUDE_SOURCE = ["The Raven (Grimm)"]
kBIBLE_REWRITE = {"Acts": "Acts of the Apostles",
                  "3 John": "Third Epistle of John",
                  "2 John": "Second Epistle of John",
                  "1 John": "First Epistle of John",
                  "Wisdom of Solomon": "Book of Wisdom"}


def read_wiki(filename, read_text=True):
    cur_title = None
    cur_id = None
    length = 0

    for ii in codecs.open(filename, 'r', encoding='utf-8'):
        if ii.startswith("<doc"):
            # Skip the first title
            if cur_title:
                if read_text:
                    yield cur_id, cur_title, text
                else:
                    yield cur_id, cur_title, length

            cur_id = int(ii.split("?curid=")[1].split('"')[0])
            try:
                cur_title = unicode(ii.split('title="')[1].split('"')[0])
            except UnicodeDecodeError:
                cur_title = unicode(ii.split('title="')[1].split('"')[0], errors='ignore')
                print("Unicode errors: %s" % (ii))
            length = 0
            text = ""
        else:
            length += len(ii)
            if read_text and not ii.startswith("</doc>"):
                text += " "
                text += ii

    if cur_title:
        if read_text:
            yield cur_id, cur_title, text
        else:
            yield cur_id, cur_title, length


def match_page(tt, answers):
    if tt in answers:
        return tt
    prefix = tt.split("/")[0]
    if prefix in answers:
        return prefix

    if "(" in tt:
        reduced = tt.split("(")[0].strip()
        if reduced in answers:
            print(unidecode(u"REDUCED: %s %s" % (tt, reduced)))
            return reduced

    for ii in ["%s (%s)" % (prefix, x) for x in kWORK_TYPES]:
        if ii in answers:
            print(unidecode("WORK: %s %s" % (tt, prefix)))
            return ii

    suffix = tt.split("/")[-1]
    for ee in kENCYCLOPEDIAS:
        if tt.startswith(ee) and suffix in answers:
            print(unidecode("ENCYC: %s %s" % (tt, suffix)))
            return suffix

    # Check to see if it's a bible chapter
    if tt.startswith("Bible "):
        if suffix in kBIBLE_REWRITE:
            suffix = kBIBLE_REWRITE[suffix]
        elif "Epistle to the %s" % suffix in answers:
            suffix = "Epistle to the %s" % suffix
        elif "Epistle to %s" % suffix in answers:
            suffix = "Epistle to %s" % suffix
        elif "Gospel of %s" % suffix in answers:
            suffix = "Gospel of %s" % suffix
        elif "Epistle of %s" % suffix in answers:
            suffix = "Epistle of %s" % suffix
        elif suffix.startswith("1 "):
            if "First Epistle to the %s" % suffix.split()[1] in answers:
                suffix = "Epistle to the %s" % suffix
            elif "First Epistle to %s" % suffix.split()[1] in answers:
                suffix = "Epistle to %s" % suffix
        elif suffix.startswith("2 "):
            if "Second Epistle to the %s" % suffix in answers:
                suffix = "Epistle to the %s" % suffix
            elif "Second Epistle to %s" % suffix in answers:
                suffix = "Epistle to %s" % suffix
        elif "Book of %s" % suffix in answers:
            suffix = "Book of %s" % suffix

        print(unidecode(u"BIBLE: %s|%s %s" %
                        (tt, suffix, str(suffix in answers))))
        if suffix in answers:
            return suffix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--question_db', type=str, default='data/questions.db')
    parser.add_argument("--wiki_location", type=str,
                        default="/Volumes/Documents/research_data/wikisource/en/*/*",
                        help="Location of wiki cache")
    parser.add_argument("--plot_location", type=str,
                        default="/Volumes/Documents/research_data/plots/*",
                        help="Location of plot summaries")
    parser.add_argument("--min_answers", type=int, default=0,
                        help="How many times does an answer need to appear to be included")
    parser.add_argument("--output_path", type=str, default="data/source",
                        help="How many pages to add to the index")
    flags = parser.parse_args()

    # Get the pages that we want to use

    answers = set(title for title, text
                  in text_iterator(False, "",
                                   True, flags.question_db,
                                   False, "",
                                   -1, min_pages=flags.min_answers))

    pages = defaultdict(str)
    for ii in glob(flags.plot_location):
        text = unidecode(gzip.open(ii, 'r').read())
        pages[ii.split("/")[-1].replace(".txt.gz", "")] = text

    print(pages.keys()[:5], pages[pages.keys()[0]][:60])

    for ii in glob(flags.wiki_location):
        for jj, tt, cc in read_wiki(ii):
            match = match_page(tt, answers)
            if match:
                pages[unidecode(match)] += "\n\n\n"
                pages[unidecode(match)] += unidecode(cc)

    for ii in pages:
        print("Writing %s" % unidecode(ii))
        o = gzip.open(u"%s/%s" % (flags.output_path, ii), 'w')
        o.write(pages[ii])
