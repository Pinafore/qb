from typing import Dict
from whoosh import index
from whoosh import query
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser, OrGroup, MultifieldParser
from whoosh.collectors import TimeLimitCollector
from whoosh import searching
import progressbar

from qanta.wikipedia.cached_wikipedia import CachedWikipedia
from qanta.util.constants import WHOOSH_WIKI_INDEX_PATH
from qanta.util.io import safe_path
from qanta import logging


log = logging.get(__name__)


class WhooshWikiIndex:
    schema = Schema(page=ID(unique=True, stored=True), content=TEXT, quiz_bowl=TEXT)

    def __init__(self, index_path=WHOOSH_WIKI_INDEX_PATH):
        self.index = index.open_dir(index_path, readonly=True)

    @classmethod
    def build(cls, documents: Dict[str, str], index_path=WHOOSH_WIKI_INDEX_PATH):
        ix = index.create_in(safe_path(index_path), cls.schema)
        writer = ix.writer()
        cw = CachedWikipedia()
        print("Building whoosh wiki index from {0} pages".format(len(documents)))
        bar = progressbar.ProgressBar()
        for p in bar(documents):
            writer.add_document(page=p, content=cw[p].content, quiz_bowl=documents[p])
        writer.commit()

    def search(self, text: str, limit: int, timelimit=2.0):
        with self.index.searcher() as searcher:
            or_group = OrGroup.factory(.9)
            parser = MultifieldParser(['content', 'quiz_bowl'], schema=self.schema, group=or_group)
            text_query = parser.parse(text)
            collector = searcher.collector(limit=limit)
            tlc = TimeLimitCollector(collector, timelimit=timelimit)
            partial = True
            try:
                searcher.search_with_collector(text_query, tlc)
                partial = False
            except searching.TimeLimit:
                pass

            # There is a bug in whoosh that makes calling len directory or indirectly fail
            # which is why we don't use list()
            results = [(r['page'], r.score) for r in tlc.results()]

            # Doing logging using partial instead of directory is required due to a mysterious race
            # condition between whoosh time limits and log.info. Its important that all of whoosh's
            # functions including search_with_collector() and tlc.results() are called before
            # logging anything
            if partial:
                log.info('Search took longer than {}s, getting partial results'.format(timelimit))

            if len(results) == 0:
                return [('<UNK_ANSWER>', 0)]

            return results

    def score_guess(self, guess, text):
        with self.index.searcher() as searcher:
            filter_query = query.Term('page', guess)
            parser = QueryParser('content', schema=self.schema)
            text_query = parser.parse(text)
            results = searcher.search(text_query, filter=filter_query)
            return results[0].score