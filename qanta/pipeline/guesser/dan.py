import luigi
from qanta.util.constants import GLOVE_WE


class GloveEmbeddings(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget(GLOVE_WE)


class DANDependencies(luigi.WrapperTask):
    def requires(self):
        yield GloveEmbeddings()
