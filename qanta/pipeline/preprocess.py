from luigi import LocalTarget, Task, WrapperTask
from qanta.pipeline.util import shell, call


class CompileCLM(Task):
    def run(self):
        shell('make clm')
        shell('touch clm/_SUCCESS')

    def output(self):
        return LocalTarget('clm/_SUCCESS')


class NLTKDownload(Task):
    def requires(self):
        CompileCLM()

    def run(self):
        shell('python3 setup.py download')
        shell('touch data/external/nltk_download_SUCCESS')

    def output(self):
        return LocalTarget('data/external/nltk_download_SUCCESS')


class GloveData(Task):
    def run(self):
        shell('mkdir -p data/external/deep')
        shell('curl http://nlp.stanford.edu/data/glove.840B.300d.zip > /tmp/glove.840B.300d.zip')
        shell('unzip /tmp/glove.840B.300d.zip -d data/external/deep')
        shell('rm /tmp/glove.840B.300d.zip')

    def output(self):
        return LocalTarget('data/external/deep/glove.840B.300d.txt')


class Wikipedia(Task):
    def requires(self):
        CompileCLM()

    def run(self):
        shell('mkdir -p data/external/wikipedia')
        shell('python3 cli.py init_wiki_cache data/external/wikipedia')
        shell('touch data/external/wikipedia/_SUCCESS')

    def output(self):
        return LocalTarget('data/external/wikipedia/_SUCCESS')


class DownloadData(WrapperTask):
    def requires(self):
        yield NLTKDownload()
        yield GloveData()
        yield Wikipedia()


class KenLM(Task):
    def requires(self):
        yield DownloadData()

    def run(self):
        shell('mkdir -p temp')
        shell('mkdir -p output')
        shell('python3 cli.py build_mentions_lm_data data/external/wikipedia /tmp/wiki_sent')
        shell('lmplz -o 5 < /tmp/wiki_sent > temp/kenlm.arpa')
        shell('build_binary temp/kenlm.arpa output/kenlm.binary')
        shell('rm /tmp/wiki_sent temp/kenlm.arpa')

    def output(self):
        return LocalTarget('output/kenlm.binary')


class WikifierInput(Task):
    def requires(self):
        yield DownloadData()

    def run(self):
        shell('rm -rf data/external/wikifier/input')
        shell('mkdir -p data/external/wikifier/input')
        shell('python3 cli.py wikify data/external/wikifier/input/')
        shell('touch data/external/wikifier/input/_SUCCESS')

    def output(self):
        return LocalTarget('data/external/wikifier/input/_SUCCESS')


class WikifierOutput(Task):
    def requires(self):
        yield WikifierInput()

    def run(self):
        shell('rm -rf data/external/wikifier/output')
        shell('mkdir -p data/external/wikifier/output')
        shell('(cd data/external/Wikifier2013 && java -Xmx10G -jar dist/wikifier-3.0-jar-with-dependencies.jar -annotateData ../wikifier/input ../wikifier/output false configs/STAND_ALONE_NO_INFERENCE.xml)')
        shell('touch data/external/wikifier/output/_SUCCESS')

    def output(self):
        return LocalTarget('data/external/wikifier/output/_SUCCESS')


class Preprocess(WrapperTask):
    def requires(self):
        yield KenLM()
        yield WikifierOutput()
