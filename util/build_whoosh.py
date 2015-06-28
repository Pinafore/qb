import argparse
from requests import ConnectionError

from whoosh.index import create_in
from whoosh.fields import TEXT, ID, Schema
from time import sleep

import wikipedia

from cached_wikipedia import CachedWikipedia
from qdb import QuestionDatabase


def text_iterator(use_wiki, wiki_location, use_qb, qb_location, limit,
                  min_pages=0):
    qdb = QuestionDatabase(qb_location)
    doc_num = 0

    cw = CachedWikipedia(wiki_location)
    pages = qdb.questions_with_pages()

    errors = {}
    for pp in sorted(pages, key=lambda k: len(pages[k]),
                     reverse=True):
        if len(pages[pp]) < min_pages:
            continue

        if use_qb:
            train_questions = [x for x in pages[pp] if x.fold == "train"]
            question_text = u"\n".join(u" ".join(x.raw_words())
                                       for x in train_questions)
        else:
            question_text = u''

        if use_wiki:
            try:
                wiki_links = cw[pp].links
            except:
                wiki_links = []

            try:
                wiki_categories = cw[pp].categories
            except:
                wiki_categories = []

            try:
                wikipedia_text = cw[pp].content + ' ' + \
                    ' '.join(wiki_links + wiki_categories)
            except wikipedia.exceptions.PageError:
                errors[pp] = "Not found"
                continue
            except wikipedia.exceptions.DisambiguationError:
                errors[pp] = "Disambiguation"
            except KeyError:
                errors[pp] = "KeyError"
            except ValueError:
                errors[pp] = "No JSON object could be decoded"
            except ConnectionError:
                print("Connection error ... ")
                errors[pp] = "Connection error"
                sleep(600)
                print("done waiting")
        else:
            wikipedia_text = u""

        total_text = wikipedia_text
        total_text += question_text

        yield pp, total_text
        doc_num += 1

        if limit > 0 and doc_num > limit:
            break

    print("ERRORS")
    print("----------------------------------------")
    for ii in errors:
        print("%s\t%s" % (ii, errors[ii]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--question_db', type=str, default='data/questions.db')
    parser.add_argument('--use_qb', default=False, action='store_true',
                        help="Use the QB data")
    parser.add_argument('--use_wiki', default=False, action='store_true',
                        help="Use wikipedia data")
    parser.add_argument("--whoosh_index", default="data/ir/whoosh",
                        help="Location of IR index")
    parser.add_argument("--wiki_location", type=str,
                        default="data/wikipedia",
                        help="Location of wiki cache")
    parser.add_argument("--min_answers", type=int, default=0,
                        help="How many times does an answer need to appear to be included")
    parser.add_argument("--max_pages", type=int, default=-1,
                        help="How many pages to add to the index")
    flags = parser.parse_args()

    schema = Schema(title=TEXT(stored=True),
                    content=TEXT(vector=True), id=ID(stored=True))
    ix = create_in(flags.whoosh_index, schema)
    writer = ix.writer()  # ix.writer(procs=4, limitmb=1024)

    errors = {}
    doc_num = 0
    for title, text in text_iterator(flags.use_wiki, flags.wiki_location,
                                     flags.use_qb, flags.question_db,
                                     flags.max_pages, flags.min_answers):

        try:
            writer.add_document(title=title, content=text, id=title)
        except IndexError:
            errors[title] = "Index error on add"
        except Exception:
            errors[title] = "Start when already in a doc"
        doc_num += 1

        if doc_num % 2500 == 0:
            print("Adding %i %s" % (doc_num, title))
            writer.commit()
            writer = ix.writer()  # ix.writer(procs=4, limitmb=1024)
    writer.commit()
