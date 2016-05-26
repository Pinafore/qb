import gzip
import re
import csv
from time import sleep

from util.cached_wikipedia import CachedWikipedia
from unidecode import unidecode

kGUESS_FIND = re.compile("(?<=guess )[^\s]*")
kLIMIT = 75

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--wiki_location', type=str, default='data/wikipedia')
    parser.add_argument('--candidates', type=str, default='')
    parser.add_argument('--countries', type=str,
                        default='data/country_list.txt')
    parser.add_argument('--infile', type=str, 
                        default='features/dev/sentence.16.vw_input')
    parser.add_argument('--outfile', type=str, 
                        default='features/dev/restricted.16.vw_input')
    flags = parser.parse_args()

    answer_candidates = set([])
    cw = CachedWikipedia(flags.wiki_location, flags.countries)

    # Read in set from csv file
    for ii in csv.DictReader(open(flags.candidates)):
        answer_candidates.add(ii["answer"])

    # Expand the set with pages one hop away in Wikipedia
    for ii in list(answer_candidates):
        for jj in [ii] + cw[ii].links[:kLIMIT]:
            answer_candidates.add(jj)
            answer_candidates.add(jj.replace(" ", "_"))
            answer_candidates.add(unidecode(jj))
            answer_candidates.add(unidecode(jj).replace(" ", "_"))

    len(answer_candidates)
    sleep(5)

    answers_found = set([])

    # Read in original training
    line = 0
    with gzip.open(flags.infile, 'rb') as infile, \
         gzip.open(flags.outfile, 'wb') as outfile:
        for ii in infile:
            guess = kGUESS_FIND.findall(ii)[0]
            if guess in answer_candidates:
                answers_found.add(guess)
                outfile.write(ii)
                line += 1
                if line % 5000 == 0:
                    print(line, guess)

    print(answers_found)
    print(len(answers_found))

