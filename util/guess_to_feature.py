# Given that features are stored in the guess database for guesses, just write
# those values out directly

import argparse
import sqlite3

from unidecode import unidecode
from fuzzywuzzy import fuzz

kMIN_TOKEN_RATIO = 80


def unicode_translation(connection):
    d = {}
    c = connection.cursor()
    query = "SELECT DISTINCT page FROM guesses;"
    c.execute(query)
    for gg, in c:
        d[unidecode(gg)] = gg
    return d


def value_lookup(connection, guesser, question, sent, token, page, trans={}):
    c = connection.cursor()
    if trans:
        query = "SELECT score FROM guesses WHERE question=? AND guesser=? " + \
            "AND sentence=? AND token=? AND (page=? or page=?)"
        c.execute(query, (question, guesser, sent, token, page, trans[page]))
    else:
        query = "SELECT score FROM guesses WHERE question=? AND guesser=? " + \
            "AND sentence=? AND token=? AND page=?"
        c.execute(query, (question, guesser, sent, token, page,))

    for score, in c:
        return score

    # If we're here, then we couldn't find an answer; let's try to find the
    # closeset match
    closest_match = ""
    score = None
    query = "SELECT page, score FROM guesses WHERE question=? " + \
        "AND guesser=? AND sentence=? AND token=?"
    c.execute(query, (question, guesser, sent, token))

    # TODO: actually get the highest score, not just the first great than
    # threshold
    for gg, ss in c:
        if fuzz.ratio(gg, page) > kMIN_TOKEN_RATIO:
            closest_match = gg
            score = ss

    if score:
        if trans:
            trans[page] = closest_match
            print(page, closest_match)
        return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--guesses', default='data/guesses.db',
                        action='store_true', help="Guesses with scores")
    parser.add_argument("--meta", default='features/dev/sentence.meta',
                        help="Metadata to order the features")
    parser.add_argument("--outfile", default="features/dev/sentence.deep.feat",
                        help="Where we write out feature file")
    parser.add_argument("--guesser", default="deep", help="Name of guesser")
    flags = parser.parse_args()

    connection = sqlite3.connect(flags.guesses)
    trans = unicode_translation(connection)

    o = open(flags.outfile, 'w')

    # Always output the same value if value not found
    not_found = "|%s %sfound:0 %sscore:0.0\n" % \
        (flags.guesser, flags.guesser, flags.guesser)

    line = 0
    for ii in open(flags.meta):
        line += 1
        qq, ss, tt, gg = ii.split(' ', 3)
        score = value_lookup(connection, flags.guesser, int(qq), int(ss),
                             int(tt), gg.strip(), trans)
        if score is None:
            print("Not found", qq, ss, tt, gg, trans[gg.strip()])
            o.write(not_found)
        else:
            o.write("|%s %sfound:1 %sscore:%f\n" %
                    (flags.guesser, flags.guesser, flags.guesser, score))
        if line % 10000 == 0:
            print(u"%i %s %s %s" % (line, qq, gg.strip(), str(score)))

    o.close()
