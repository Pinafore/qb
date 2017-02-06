import gzip
import zlib
import os

from qanta import logging
from qanta.wikipedia.cached_wikipedia import CachedWikipedia
from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.util.environment import data_path
from qanta.util.constants import COUNTRY_LIST_PATH

log = logging.get(__name__)


def text_iterator(use_wiki, wiki_location,
                  use_qb, qb_location,
                  use_source, source_location,
                  limit=-1,
                  min_pages=0, country_list=COUNTRY_LIST_PATH):
    if isinstance(qb_location, str):
        qdb = QuestionDatabase(qb_location)
    else:
        qdb = qb_location
    doc_num = 0

    cw = CachedWikipedia(wiki_location, data_path(country_list))
    pages = qdb.questions_with_pages()

    for pp in sorted(pages, key=lambda k: len(pages[k]), reverse=True):
        # This bit of code needs to line up with the logic in qdb.py
        # to have the same logic as the page_by_count function
        if len(pages[pp]) < min_pages:
            continue

        if use_qb:
            train_questions = [x for x in pages[pp] if x.fold == "train"]
            question_text = u"\n".join(u" ".join(x.raw_words())
                                       for x in train_questions)
        else:
            question_text = u''

        if use_source:
            filename = '%s/%s' % (source_location, pp)
            if os.path.isfile(filename):
                try:
                    with gzip.open(filename, 'rb') as f:
                        source_text = f.read()
                except zlib.error:
                    log.info("Error reading %s" % filename)
                    source_text = ''
            else:
                source_text = ''
        else:
            source_text = u''

        if use_wiki:
            wikipedia_text = cw[pp].content
        else:
            wikipedia_text = u""

        total_text = wikipedia_text
        total_text += "\n"
        total_text += question_text
        total_text += "\n"
        total_text += str(source_text)

        yield pp, total_text
        doc_num += 1

        if 0 < limit < doc_num:
            break
