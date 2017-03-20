from typing import Dict
from elasticsearch_dsl import DocType, Text, Keyword, Search
from elasticsearch_dsl.connections import connections
import progressbar

from qanta.wikipedia.cached_wikipedia import CachedWikipedia


connections.create_connection(hosts=['localhost'])


class Answer(DocType):
    page = Text(fields={'raw': Keyword()})
    wiki_content = Text()
    qb_content = Text()

    class Meta:
        index = 'qb'


class ElasticSearchIndex:
    @staticmethod
    def build(documents: Dict[str, str]):
        Answer.init()
        cw = CachedWikipedia()
        bar = progressbar.ProgressBar()
        for page in bar(documents):
            answer = Answer(page=page, wiki_content=cw[page].content, qb_content=documents[page])
            answer.save()

    def search(self, text: str):
        s = Search(index='qb').query(
            'multi_match', query=text, fields=['wiki_content', 'qb_content'])
        results = s.execute()
        return [(r.page, r.meta.score) for r in results]
