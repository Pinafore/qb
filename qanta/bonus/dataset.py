import re
import os
import glob
import pickle
import itertools
from tqdm import tqdm
from typing import List, Dict, Iterable, Tuple
from collections import defaultdict
from bs4 import BeautifulSoup

from qanta.util import constants as c
from qanta.util.multiprocess import _multiprocess
from qanta.util.environment import BONUS_QUESTION_DB, BONUS_QUESTION_PKL, NAQT_QBML_DIR


class BonusQuestion:

    def __init__(self, qnum, texts, pages, answers, leadin=None, fold=None):
        self.qnum = qnum
        # assert len(texts) == 3 and len(pages) == 3 and len(answers) == 3
        self.texts = texts
        self.pages = pages
        self.answers = answers
        self.leadin = leadin
        self.fold = fold
        assert len(pages) == len(texts)

    def __repr__(self):
        s = '<BonusQuestion qnum={} fold={} \n' + \
            'leadin: {}\n'
        s += ' '.join([str(i) + ': page={}, text={}...\n' for i in
            range(len(self.pages))])
        values = [self.qnum, self.fold, self.leadin]
        values += itertools.chain(*list(zip(self.pages, self.texts)))
        return s.format(*values)


class BonusQuestionDatabase:
    
    def __init__(self, location=BONUS_QUESTION_PKL):
        if os.path.isfile(BONUS_QUESTION_PKL):
            with open(location, 'rb') as f:
                self.questions = pickle.load(f)
        else:
            self.questions = self.load_qbml(NAQT_QBML_DIR, location)
        self.questions = {x.qnum: x for x in self.questions}

    def _process_question(self, qnum, qstr):
        '''
    
        For 10 points each--answer these questions about the U.S. Supreme Court's
        1995-96 term.
    
        A.      These two justices, considered the court's center, issued fewer
        dissents than any other justices.
    
        answer: Anthony M. _Kennedy_, Sandra Day _O'Connor_
    
        B.      Considered the court's most liberal justice, he dissented in 19 of
        the courts 41 contested rulings.
    
        answer: John Paul _Stevens_
        '''
        q = [x for x in qstr.strip().split('\n') if len(x)]
        leadin = q[0].strip()
        texts = []
        answers = []
        i = 1
        while i + 1 < len(q):
            if not re.match("[A-Z]\.\t*", q[i]):
                return None
            texts.append(q[i][2:].strip())
            i += 1
            if not re.match("[Aa]nswer:\t*", q[i]):
                return None
            answers.append(q[i][8:].strip())
            i += 1
            # don't deal with questions with multiple answers
            # while i < len(q) and not re.match("[A-Z].\t*", q[i]):
            #     answers[-1].append(q[i].strip())
            #     i += 1
        q = BonusQuestion(qnum, texts, answers, answers, leadin=leadin)
        return q

    def load_qbml(self, dir, pkl_dir):
        qbml_dirs = glob.glob(dir + '*.qbml')
        bonus_questions = []
        for qbml_dir in tqdm(qbml_dirs):
            with open(qbml_dir) as f:
                soup = BeautifulSoup(f.read(), 'xml')
            questions = soup.find_all('QUESTION')
            bonus_qs = [(q.attrs['ID'], next(q.children).title()) for q in questions if
                    q.attrs['KIND'] == 'BONUS']
            bonus_qs = _multiprocess(self._process_question, bonus_qs, progress=False)
            bonus_qs = [x for x in bonus_qs if x is not None]
            bonus_questions += bonus_qs
        with open(pkl_dir, 'wb') as f:
            pickle.dump(bonus_questions, f)
        return bonus_questions

    def all_questions(self) -> Dict[int, BonusQuestion]:
        return self.questions

class BonusQuestionDatabaseFromSQL:

    def __init__(self, location=BONUS_QUESTION_DB):
        self._conn = sqlite3.connect(location)
    
    def all_questions(self) -> Dict[int, BonusQuestion]:
        questions = {}
        c = self._conn.cursor()
        c.execute('select * from text where page != ""')
        question_parts = defaultdict(dict)
        for qid, number, _, text, page, answer, _ in c:
            question_parts[int(qid)][int(number)] = (text, page, answer)
        bonus_questions = dict()
        for qnum, parts in question_parts.items():
            if not set(parts.keys()) == {0,1,2}:
                # log.info('skipping {}, missing question parts'.format(qnum))
                continue
            # transpose
            parts = list(zip(*[parts[i] for i in [0,1,2]]))
            bonus_questions[qnum] = BonusQuestion(qnum, parts[0], parts[1], parts[2])

        c = self._conn.cursor()
        c.execute('select * from questions')
        extra_parts = dict()
        for qnum, tour, leadin, _, fold in c:
            qnum = int(qnum)
            if qnum not in bonus_questions:
                continue
            bonus_questions[qnum].leadin = leadin
            bonus_questions[qnum].fold = fold
        return bonus_questions
