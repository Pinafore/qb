from luigi import Task, LocalTarget
from clm.lm_wrapper import build_clm
from qanta.util import constants as c
from qanta.pipeline.preprocess import DownloadData
from qanta.util.io import shell
from qanta.features import mentions
from qanta.features.stats import compute_question_stats
from qanta.util.environment import QB_QUESTION_DB


class BuildClm(Task):
    def requires(self):
        yield DownloadData()

    def output(self):
        return LocalTarget(c.CLM_TARGET)

    def run(self):
        build_clm()


class KenLM(Task):
    def requires(self):
        yield DownloadData()

    def run(self):
        shell('mkdir -p temp')
        shell('mkdir -p output')
        mentions.build_lm_data('/tmp/wikipedia_sentences')
        shell('lmplz -o 5 < /tmp/wikipedia_sentences > /tmp/kenlm.arpa')
        shell('build_binary /tmp/kenlm.arpa {}'.format(c.KEN_LM))
        shell('rm /tmp/wikipedia_sentences /tmp/kenlm.arpa')

    def output(self):
        return LocalTarget(c.KEN_LM)


class ComputeParagraphStats(Task):
    def output(self):
        return LocalTarget(c.SENTENCE_STATS)

    def run(self):
        compute_question_stats(QB_QUESTION_DB)


