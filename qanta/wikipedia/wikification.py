from unidecode import unidecode

from qanta.util.environment import QB_QUESTION_DB
from qanta.util.constants import MIN_APPEARANCES
from qanta.util.qdb import QuestionDatabase


def wikify(output_directory):
    database = QuestionDatabase(QB_QUESTION_DB)
    pages = database.questions_with_pages()

    total = 0
    for p in pages:
        if len(pages[p]) >= MIN_APPEARANCES:
            print(p, len(pages[p]))
            for q in pages[p]:
                total += 1
                for sentence, word, text in q.partials():
                    sentence -= 1
                    with open("%s/%i-%i.txt" % (output_directory, q.qnum, sentence),
                              'w') as output:
                        output.write("%s\n" % unidecode(text[sentence]))
    print(total)
