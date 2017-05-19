import luigi
from luigi import LocalTarget, Task, WrapperTask
from qanta.util import constants as c
from qanta.util.io import call, shell, make_dirs
from qanta.reporting import performance


@luigi.Task.event_handler(luigi.Event.PROCESSING_TIME)
def get_execution_time(self, processing_time):
    self.execution_time = processing_time


# All the code below needs to be integrated with new buzzer pipeline/rewritten
class Summary(Task):
    fold = luigi.Parameter()

    def output(self):
        return LocalTarget('output/summary/{0}.json'.format(self.fold))

    def run(self):
        make_dirs('output/summary/')
        call([
            'python3',
            'qanta/reporting/performance.py',
            'generate',
            c.PRED_TARGET.format(self.fold),
            c.META_TARGET.format(self.fold),
            'output/summary/{0}.json'.format(self.fold)
        ])


class AllSummaries(WrapperTask):
    def requires(self):
        for fold in c.BUZZ_FOLDS:
            yield Summary(fold=fold)


class ConcatReports(Task):
    resources = {'report_write': 1}

    def requires(self):
        yield AllSummaries()

    def output(self):
        return LocalTarget('output/reporting/report.pdf')

    def run(self):
        shell('pdftk output/reporting/*.pdf cat output /tmp/report.pdf')
        shell('mv /tmp/report.pdf output/reporting/report.pdf')


class PerformancePlot(Task):
    def requires(self):
        yield Summary(fold='test')

    def output(self):
        return LocalTarget('output/summary/performance.png')

    def run(self):
        performance.plot_summary(False, 'output/summary/', 'output/summary/performance.png')


class All(WrapperTask):
    def requires(self):
        yield ConcatReports()
        yield PerformancePlot()
