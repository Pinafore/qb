from luigi import LocalTarget, Task, WrapperTask, ExternalTask
from qanta.util.io import shell
from qanta.util.constants import ALL_WIKI_REDIRECTS, WIKI_DUMP_REDIRECT_PICKLE
from qanta.wikipedia.cached_wikipedia import create_wikipedia_redirect_pickle


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
        shell('aws s3 cp {} data/external/parsed-wiki.tar.lz4'.format(s3_location))
        shell('lz4 -d data/external/parsed-wiki.tar.lz4 | tar -x -C data/external/')
        shell('touch data/external/parsed-wiki_SUCCESS')

    def output(self):
        return [LocalTarget('data/external/parsed-wiki_SUCCESS'), LocalTarget('data/external/parsed-wiki/')]


class WikipediaTitles(Task):
    def run(self):
        pass

    def output(self):
        return LocalTarget('data/external/wikipedia-titles.pickle')


class BuildWikipediaCache(Task):
    def requires(self):
        yield WikipediaDumps()


class Wikipedia(Task):
    def run(self):
        shell('mkdir -p data/external/wikipedia')
        shell('python3 cli.py init_wiki_cache data/external/wikipedia')
        shell('touch data/external/wikipedia_SUCCESS')

    def output(self):
        return LocalTarget('data/external/wikipedia_SUCCESS')


class DownloadData(WrapperTask):
    def requires(self):
        yield CodeCompile()
        yield Wikipedia()
