import os

from luigi import LocalTarget, Task, WrapperTask, ExternalTask

from qanta.util.io import shell, safe_path
from qanta.util.constants import (
    ALL_WIKI_REDIRECTS, WIKI_DUMP_REDIRECT_PICKLE, WIKI_TITLES_PICKLE, WIKI_INSTANCE_OF_PICKLE, WIKI_LOOKUP_PATH
)
from qanta.util.environment import is_aws_authenticated
from qanta.wikipedia.wikidata import create_instance_of_map
from qanta.wikipedia.cached_wikipedia import (
    create_wikipedia_redirect_pickle, create_wikipedia_title_pickle, create_wikipedia_cache
)


WIKIDATA_CLAIMS = 'data/external/wikidata-claims_instance-of.jsonl'


class NLTKDownload(ExternalTask):
    """
    To complete this task run `python setup.py download`
    """
    def output(self):
        return LocalTarget('data/external/nltk_download_SUCCESS')


class WikipediaRawRedirects(Task):
    def run(self):
        safe_path(ALL_WIKI_REDIRECTS)
        if is_aws_authenticated():
            s3_location = 's3://pinafore-us-west-2/public/wiki_redirects.csv'
            shell('aws s3 cp {} {}'.format(s3_location, ALL_WIKI_REDIRECTS))
        else:
            https_location = 'https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/public/wiki_redirects.csv'
            shell('wget -O {} {}'.format(ALL_WIKI_REDIRECTS, https_location))

    def output(self):
        return LocalTarget(ALL_WIKI_REDIRECTS)


class WikipediaRedirectPickle(Task):
    def requires(self):
        yield WikipediaRawRedirects()

    def run(self):
        create_wikipedia_redirect_pickle(ALL_WIKI_REDIRECTS, WIKI_DUMP_REDIRECT_PICKLE)

    def output(self):
        yield LocalTarget(WIKI_DUMP_REDIRECT_PICKLE)


class WikipediaDumps(Task):
    def run(self):
        archive = safe_path('data/external/wikipedia/parsed-wiki.tar.lz4')
        if is_aws_authenticated():
            s3_location = f's3://pinafore-us-west-2/public/parsed-wiki.tar.lz4'
            shell(f'aws s3 cp {s3_location} {archive}')
        else:
            https_location = 'https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/public/parsed-wiki.tar.lz4'
            shell(f'wget -O {archive} {https_location}')

        shell(f'lz4 -d {archive} | tar -x -C data/external/wikipedia/')
        shell(f'rm {archive}')
        shell('touch data/external/wikipedia/parsed-wiki_SUCCESS')

    def output(self):
        return [
            LocalTarget('data/external/wikipedia/parsed-wiki_SUCCESS'),
            LocalTarget('data/external/wikipedia/parsed-wiki/')
        ]


class WikipediaTitles(Task):
    def requires(self):
        yield WikipediaDumps()

    def run(self):
        # Spark needs an absolute path for local files
        dump_path = os.path.abspath('data/external/wikipedia/parsed-wiki/*/*')
        create_wikipedia_title_pickle(dump_path, WIKI_TITLES_PICKLE)

    def output(self):
        return LocalTarget(WIKI_TITLES_PICKLE)


class BuildWikipediaCache(Task):
    def requires(self):
        yield WikipediaDumps()

    def run(self):
        create_wikipedia_cache()

    def output(self):
        return [
            LocalTarget(WIKI_LOOKUP_PATH),
        ]


class WikidataInstanceOfDump(Task):
    def run(self):
        s3_location = 's3://entilzha-us-west-2/wikidata/wikidata-claims_instance-of.jsonl'
        shell('aws s3 cp {} {}'.format(s3_location, WIKIDATA_CLAIMS))

    def output(self):
        return LocalTarget(WIKIDATA_CLAIMS)


class WikidataInstanceOfPickle(Task):
    def requires(self):
        yield WikidataInstanceOfDump()

    def run(self):
        create_instance_of_map(
            WIKIDATA_CLAIMS,
            WIKI_INSTANCE_OF_PICKLE
        )

    def output(self):
        return LocalTarget(
            WIKI_INSTANCE_OF_PICKLE
        )


class DownloadData(WrapperTask):
    def requires(self):
        yield NLTKDownload()
        yield BuildWikipediaCache()
        yield WikipediaTitles()
        yield WikipediaRedirectPickle()
