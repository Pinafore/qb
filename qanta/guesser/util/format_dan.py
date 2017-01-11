import pickle

from qanta.preprocess import format_guess, Preprocessor
from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.util.io import safe_open
from qanta.util.constants import MIN_APPEARANCES, DEEP_VOCAB_TARGET
from qanta.util.environment import QB_QUESTION_DB


def preprocess():
    pp = Preprocessor()
    db = QuestionDatabase(QB_QUESTION_DB)

    pages = set(db.page_by_count(MIN_APPEARANCES, True))
    print(len(pages))
    folds = ['train', 'test', 'devtest', 'dev']
    for fold in folds:
        allqs = db.query('from questions where page != "" and fold == ?', (fold,))
        print(fold, len(allqs))
        proc_fold = []
        for i, key in enumerate(allqs):
            q = allqs[key]
            if q.page in pages:
                qs = {}
                for index in q.text:
                    qs[index] = pp.preprocess_question(q.text[index])
                ans = format_guess(q.page)
                answer = pp.convert_to_indices(ans)
                proc_fold.append((qs, answer))
            if i % 5000 == 0:
                print('done with ', i)

        print(fold, len(proc_fold))
        with safe_open('output/deep/' + fold, 'wb') as f:
            pickle.dump(proc_fold, f, protocol=pickle.HIGHEST_PROTOCOL)

    with safe_open(DEEP_VOCAB_TARGET, 'wb') as f:
        pickle.dump((pp.vocab, pp.vdict), f, protocol=pickle.HIGHEST_PROTOCOL)
