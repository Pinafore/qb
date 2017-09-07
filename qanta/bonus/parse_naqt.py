import re
import glob
import pickle
from bs4 import BeautifulSoup
from multiprocessing import Pool

from qanta.util.multiprocess import _multiprocess
from qanta.datasets.quiz_bowl import BonusQuestion

def _process_question(q):
    # q is a bs4.element.Tag
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
    q = [x for x in q.strip().split('\n') if len(x)]
    lead_in = q[0]
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
    return {'lead_in': lead_in, 'texts': texts, 'answers': answers}

def main():
    qbml_dirs = glob.glob('data/internal/naqt_qbml/*.qbml')
    bonus_questions = []
    for qbml_dir in qbml_dirs:
        print(qbml_dir)
        with open(qbml_dir) as f:
            soup = BeautifulSoup(f.read(), 'xml')
        questions = soup.find_all('QUESTION')
        bonus_qs = [[next(q.children).title()] for q in questions if
                q.attrs['KIND'] == 'BONUS']
        pool = Pool(8)
        bonus_qs = _multiprocess(_process_question, bonus_qs, multi=False)
        bonus_qs = [x for x in bonus_qs if x is not None]
        print(len(bonus_qs))
        bonus_questions += bonus_qs
    with open('data/internal/naqt_qbml.pkl', 'wb') as f:
        pickle.dump(bonus_questions, f)
    print(len(bonus_questions))

if __name__ == '__main__':
    main()
