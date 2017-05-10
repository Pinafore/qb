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


class CreateTestQuestions(Task):
    def output(self):
        return LocalTarget(safe_path(EXPO_QUESTIONS))

    def run(self):
        db = QuestionDatabase(QB_QUESTION_DB)
        questions = db.all_questions()
        with open(safe_path(EXPO_QUESTIONS), 'w', newline='') as f:
            f.write('id,answer,sent,text\n')
            writer = csv.writer(f, delimiter=',')
            for q in questions.values():
                if q.fold != 'test':
                    continue
                max_sent = max(q.text.keys())
                for i in range(max_sent + 1):
                    writer.writerow([q.qnum, format_guess(q.page), i, q.text[i]])


class Prerequisites(ExternalTask):
    fold = luigi.Parameter()

    def output(self):
        return [LocalTarget(PRED_TARGET.format(self.fold)),
                LocalTarget(META_TARGET.format(self.fold))]


class GenerateExpo(Task):
    fold = luigi.Parameter()

    def requires(self):
        yield Prerequisites(fold=self.fold)

    def output(self):
        return [LocalTarget(EXPO_BUZZ.format(self.fold)),
                LocalTarget(EXPO_FINAL.format(self.fold))]

    def run(self):
        db = QuestionDatabase(QB_QUESTION_DB)
        data = load_data(PRED_TARGET.format(self.fold),
                         META_TARGET.format(self.fold), db)
        audit_data = load_audit(VW_AUDIT.format(self.fold), META_TARGET.format(self.fold))
        buzz_file = open(safe_path(EXPO_BUZZ.format(self.fold)), 'w', newline='')
        buzz_file.write('question,sentence,word,page,evidence,final,weight\n')
        buzz_writer = csv.writer(buzz_file, delimiter=',')

        final_file = open(safe_path(EXPO_FINAL.format(self.fold)), 'w', newline='')
        final_file.write('question,answer\n')
        final_writer = csv.writer(final_file, delimiter=',')

        for qnum, lines in data:
            final_sentence, final_token, final_guess = find_final(lines)
            if final_sentence == -1 and final_token == -1:
                final_writer.writerow([qnum, final_guess])

            for l in lines:
                i = 0
                is_final = False
                if l.sentence == final_sentence and l.token == final_token:
                    final_writer.writerow([qnum, l.guess])
                    is_final = True

                for g in l.all_guesses:
                    evidence = audit_data[(l.question, l.sentence, l.token, g.guess)]
                    buzz_writer.writerow([
                        l.question, l.sentence, l.token, g.guess, evidence,
                        int(is_final and g.guess == l.guess), g.score
                    ])
                    i += 1
                    if i > 4:
                        break
        buzz_file.close()
        final_file.close()


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
        yield GenerateExpoBuzzer(fold='test')
        yield CreateTestQuestions()
