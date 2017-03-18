from qanta import logging
from qanta.util.environment import QB_QUESTION_DB
from qanta.config import conf
from qanta.datasets.quiz_bowl import QuestionDatabase

log = logging.get(__name__)


def wikify(output_directory):
    database = QuestionDatabase(QB_QUESTION_DB)
    pages = database.questions_with_pages()

    total = 0
    for p in pages:
        if len(pages[p]) >= conf['wikifier']['min_appearances']:
            log.info('{} {}'.format(p, len(pages[p])))
            for q in pages[p]:
                total += 1
                for sentence, word, text in q.partials():
                    sentence -= 1
                    with open("%s/%i-%i.txt" % (output_directory, q.qnum, sentence),
                              'w') as output:
                        output.write("%s\n" % text[sentence])
    log.info(str(total))
