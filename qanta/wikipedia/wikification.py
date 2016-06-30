from unidecode import unidecode
import argparse
from collections import defaultdict

from qanta.util.environment import QB_QUESTION_DB
from qanta.util.qdb import QuestionDatabase


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--min_pages', type=int, default=4)
    parser.add_argument("--output_directory", type=str,
                        default="output/wikifier/data/input/",
                        help="Where we write output file")
    parser.add_argument("--database", type=int, default=1, help="1 to load questions, 0 for no")

    flags = parser.parse_args()

    database = QuestionDatabase(QB_QUESTION_DB)
    

    if flags.database:
        pages = database.questions_with_pages()
    else:
        pages = defaultdict(set)

    total = 0
    for pp in pages:
        if len(pages[pp]) >= flags.min_pages:
            print(pp, len(pages[pp]))
            for qq in pages[pp]:
                total += 1
                for sentence, word, text in qq.partials():
                    sentence -= 1
                    with open("%s/%i-%i.txt" % (flags.output_directory, qq.qnum, sentence),
                              'w') as output:
                        output.write("%s\n" % unidecode(text[sentence]))
    print(total)

if __name__ == "__main__":
    main()
