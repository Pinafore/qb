import csv
from argparse import Namespace

import luigi
from luigi import LocalTarget, Task, ExternalTask, WrapperTask

from qanta.config import conf
from qanta.reporting.performance import load_data, load_audit
from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.preprocess import format_guess
from qanta.util.io import safe_path
from qanta.util.environment import QB_QUESTION_DB
from qanta.util.constants import (PRED_TARGET, META_TARGET, EXPO_BUZZ, EXPO_FINAL, VW_AUDIT,
                                  EXPO_QUESTIONS)
import qanta.buzzer.test

def find_final(lines):
    for l in lines:
        if l.buzz:
            return l.sentence, l.token, l.guess
    return -1, -1, lines[-1].guess


class CreateQuestions(Task):
    fold = luigi.Parameter()

    def output(self):
        return LocalTarget(safe_path(EXPO_QUESTIONS.format(self.fold)))

    def run(self):
        db = QuestionDatabase(QB_QUESTION_DB)
        questions = db.all_questions()
        with open(safe_path(EXPO_QUESTIONS.format(self.fold)), 'w', newline='') as f:
            f.write('id,answer,sent,text\n')
            writer = csv.writer(f, delimiter=',')
            for q in questions.values():
                if q.fold != self.fold:
                    continue
                max_sent = max(q.text.keys())
                for i in range(max_sent + 1):
                    writer.writerow([q.qnum, format_guess(q.page), i, q.text[i]])


class Prerequisites(ExternalTask):
    fold = luigi.Parameter()

    def output(self):
        return [LocalTarget(PRED_TARGET.format(self.fold)),
                LocalTarget(META_TARGET.format(self.fold))]


class GenerateExpoBuzzer(Task):
    fold = luigi.Parameter()

    def requires(self):
        yield Prerequisites(fold=self.fold)

    def output(self):
        return [LocalTarget(EXPO_BUZZ.format(self.fold)),
                LocalTarget(EXPO_FINAL.format(self.fold))]

    def run(self):
        args = Namespace
        args.config = conf['buzzer']['config']
        args.fold = self.fold
        qanta.buzzer.test.generate(args)

class GenerateExpoBuzzer(Task):
    fold = luigi.Parameter()

    def requires(self):
        yield Prerequisites(fold=self.fold)

    def output(self):
        return [LocalTarget(EXPO_BUZZ.format(self.fold)),
                LocalTarget(EXPO_FINAL.format(self.fold))]

    def run(self):
        args = Namespace
        args.config = conf['buzzer']['config']
        args.fold = self.fold
        qanta.buzzer.test.generate(args)

class AllExpo(WrapperTask):
    def requires(self):
        yield GenerateExpoBuzzer(fold='expo')
        yield CreateQuestions(fold='expo')
        yield GenerateExpoBuzzer(fold='test')
        yield CreateQuestions(fold='test')
