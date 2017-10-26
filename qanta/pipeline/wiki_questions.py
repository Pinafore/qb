import luigi
from luigi import LocalTarget, Task
from qanta.config import conf
from qanta.guesser.util.wiki_questions import generate_domain_classifier_data, get_best_wiki_questions
from qanta.util import constants as c
from qanta.util.io import call


class WikiQuestionsVW(Task):
    def output(self):
        return [
            LocalTarget(c.DOMAIN_TARGET_PREFIX + '0'),
            LocalTarget(c.DOMAIN_TARGET_PREFIX + '1')
        ]

    def run(self):
        generate_domain_classifier_data()


class TrainDomainVW(Task):
    group_num = luigi.IntParameter()

    def requires(self):
        yield WikiQuestionsVW()

    def run(self):
        call([
            'vw',
            '-d', c.DOMAIN_TARGET_PREFIX + str(self.group_num),
            '-k',
            '-q', 'tt',
            '-b', '28',
            '--loss_function', 'logistic',
            '-f', c.DOMAIN_MODEL_FORMAT.format(self.group_num)
        ])

    def output(self):
        return LocalTarget(c.DOMAIN_MODEL_FORMAT.format(self.group_num))


class LabelWikiQuestionsVW(Task):
    model_num = luigi.IntParameter()
    data_num = luigi.IntParameter()

    def requires(self):
        yield TrainDomainVW(group_num=self.model_num)
        yield WikiQuestionsVW()

    def run(self):
        call([
            'vw',
            '-i', c.DOMAIN_MODEL_FORMAT.format(self.model_num),
            '-t',
            '-k',
            '--loss_function', 'logistic',
            '-d', c.DOMAIN_TARGET_PREFIX + str(self.data_num),
            '-f', c.DOMAIN_MODEL_FORMAT.format(self.model_num),
            '-p', c.DOMAIN_PREDICTIONS_PREFIX + str(self.data_num),
        ])

    def output(self):
        return LocalTarget(c.DOMAIN_PREDICTIONS_PREFIX + str(self.data_num))


class SelectWikiQuestions(Task):
    def requires(self):
        if conf['wiki_data_frac'] > 0:
            for group in (0, 1):
                return LabelWikiQuestionsVW(data_num=group, model_num=1-group)
        else:
            return []

    def run(self):
        if conf['wiki_data_frac'] > 0:
            get_best_wiki_questions(frac_questions=conf['wiki_data_frac'])
        else:
            from pathlib import Path
            Path(c.DOMAIN_OUTPUT.format('frac=0')).touch()

    def output(self):
        return LocalTarget(c.DOMAIN_OUTPUT.format('frac=' + str(conf['wiki_data_frac'])))
