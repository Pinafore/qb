import os

from luigi import LocalTarget, Task, WrapperTask, ExternalTask

from qanta.util.io import shell
from qanta.util.constants import ALL_WIKI_REDIRECTS, WIKI_DUMP_REDIRECT_PICKLE, WIKI_TITLES_PICKLE
from qanta.wikipedia.cached_wikipedia import (
    create_wikipedia_redirect_pickle, create_wikipedia_title_pickle, create_wikipedia_cache
)


class NLTKDownload(ExternalTask):
    """
    To complete this task run `python setup.py download`
    """
    def output(self):
        return LocalTarget('data/external/nltk_download_SUCCESS')


class CompileCLM(ExternalTask):
    """
    To complete this task run `make clm`
    """
    def output(self):
        return LocalTarget('clm/_SUCCESS')


class CodeCompile(WrapperTask):
    def requires(self):
        yield NLTKDownload()
        yield CompileCLM()


class WikipediaRawRedirects(Task):
    def run(self):
        s3_location = 's3://pinafore-us-west-2/public/wiki_redirects.csv'
        shell('aws s3 cp {} {}'.format(s3_location, ALL_WIKI_REDIRECTS))

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
        s3_location = 's3://pinafore-us-west-2/public/wikipedia-dumps/parsed-wiki.tar.lz4'
        shell('aws s3 cp {} data/external/wikipedia/parsed-wiki.tar.lz4'.format(s3_location))
        shell('lz4 -d data/external/wikipedia/parsed-wiki.tar.lz4 | tar -x -C data/external/wikipedia/')
        shell('rm data/external/wikipedia/parsed-wiki.tar.lz4')
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
        dump_path = os.path.abspath('data/external/wikipedia/parsed-wiki/*/*')
        create_wikipedia_cache(dump_path)
        shell('touch data/external/wikipedia/cache_SUCCESS')

    def output(self):
        return [
            LocalTarget('data/external/wikipedia/cache_SUCCESS'),
            LocalTarget('data/external/wikipedia/pages/')
        ]


class DownloadData(WrapperTask):
    def requires(self):
        yield CodeCompile()
        yield BuildWikipediaCache()
        yield WikipediaTitles()
        yield WikipediaRedirectPickle()
