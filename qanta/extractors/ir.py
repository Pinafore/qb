from abc import ABCMeta, abstractmethod

from whoosh import index
from whoosh import query
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
import progressbar

from qanta.wikipedia.cached_wikipedia import CachedWikipedia
from qanta.extractors.abstract import AbstractFeatureExtractor
from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.util.constants import WHOOSH_WIKI_INDEX_PATH, COUNTRY_LIST_PATH, MAX_APPEARANCES
from qanta.util.environment import QB_WIKI_LOCATION, QB_QUESTION_DB


class IrExtractor(AbstractFeatureExtractor):
    def __init__(self):
        super(IrExtractor, self).__init__()
        self.wiki_index = WikiIndex()

    @property
    def name(self):
        return 'ir'

    def score_guesses(self, guesses, text):
        pass

    def vw_from_score(self, results):
        pass

    def text_guess(self, text):
        return dict(self.wiki_index.search(text))


class Index(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def build(cls):
        pass


class WikiIndex(Index):
    schema = Schema(page=ID(unique=True, stored=True), content=TEXT)

    @classmethod
    def build(cls):
        ix = index.create_in(WHOOSH_WIKI_INDEX_PATH, cls.schema)
        writer = ix.writer()
        cw = CachedWikipedia(QB_WIKI_LOCATION, COUNTRY_LIST_PATH)
        qdb = QuestionDatabase(QB_QUESTION_DB)
        questions = qdb.questions_with_pages()
        pages = [page for page, questions in questions if len(questions) < MAX_APPEARANCES]
        pages = list(qdb.get_all_pages(exclude_test=True))
        print("Building whoosh wiki index from {0} pages".format(len(pages)))
        bar = progressbar.ProgressBar()
        for p in bar(pages):
            writer.add_document(page=p, content=cw[p].content)
        writer.commit()

    def __init__(self):
        self.index = index.open_dir(WHOOSH_WIKI_INDEX_PATH, readonly=True)

    def search(self, text, limit=10):
        with self.index.searcher() as searcher:
            parser = QueryParser('content', schema=self.schema)
            text_query = parser.parse(text)
            results = searcher.search(text_query, limit=limit)
            return [(r['page'], r.score) for r in results]

    def score_guess(self, guess, text):
        with self.index.searcher() as searcher:
            filter_query = query.Term('page', guess)
            parser = QueryParser('content', schema=self.schema)
            text_query = parser.parse(text)
            results = searcher.search(text_query, filter=filter_query)
            return results[0].score
