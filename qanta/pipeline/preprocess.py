from luigi import LocalTarget, Task, WrapperTask, ExternalTask
from qanta.util.io import shell


class NLTKDownload(ExternalTask):
    def output(self):
        return LocalTarget('data/external/nltk_download_SUCCESS')


class CompileCLM(ExternalTask):
    def output(self):
        return LocalTarget('clm/_SUCCESS')


class Wikipedia(Task):
    def requires(self):
        CompileCLM()

    def run(self):
        shell('mkdir -p data/external/wikipedia')
        shell('python3 cli.py init_wiki_cache data/external/wikipedia')
        shell('touch data/external/wikipedia_SUCCESS')

    def output(self):
        return LocalTarget('data/external/wikipedia_SUCCESS')


class DownloadData(WrapperTask):
    def requires(self):
        yield NLTKDownload()
        yield Wikipedia()
