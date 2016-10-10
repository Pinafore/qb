import luigi
from luigi import LocalTarget, Task, ExternalTask

from qanta.reporting.performance import load_data
from qanta.util.io import safe_open
from qanta.util.qdb import QuestionDatabase
from qanta.util.environment import QB_QUESTION_DB
from qanta.util.constants import PRED_TARGET, META_TARGET, EXPO_BUZZ, EXPO_FINAL


class Prequisites(ExternalTask):
    fold = luigi.Parameter()
    weight = luigi.IntParameter()

    def output(self):
        return [LocalTarget(PRED_TARGET.format(self.fold, self.weight)),
                LocalTarget(META_TARGET.format(self.fold, self.weight))]


class GenerateExpo(Task):
    fold = luigi.Parameter()
    weight = luigi.IntParameter()

    def requires(self):
        yield Prequisites(fold=self.fold, weight=self.weight)

    def output(self):
        return [LocalTarget(EXPO_BUZZ.format(self.fold, self.weight)),
                LocalTarget(EXPO_FINAL.format(self.fold, self.weight))]

    def run(self):
        db = QuestionDatabase(QB_QUESTION_DB)
        data = load_data(PRED_TARGET.format(self.fold, self.weight),
                         META_TARGET.format(self.fold, self.weight), db)
        buzz_file = safe_open(EXPO_BUZZ.format(self.fold, self.weight), 'w')
        buzz_file.write('question,sentence,word,page,evidence,final,weight\n')

        final_file = safe_open(EXPO_FINAL.format(self.fold, self.weight), 'w')
        final_file.write('question,answer\n')
        for qnum, lines in data:
            for l in lines:
                i = 0
                final_file.write('{},{}\n'.format(qnum, l.guess))
                for g in l.all_guesses:
                    o = ','.join([
                        str(l.question), str(l.sentence), str(l.token),
                        g.guess, '', str(int(g.score > 0)), str(g.score)])
                    buzz_file.write(o)
                    buzz_file.write('\n')
                    i += 1
                    if i > 4:
                        break
        buzz_file.close()
        final_file.close()
