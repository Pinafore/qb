from unidecode import unidecode
import argparse
from util.qdb import QuestionDatabase

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--database', type=str, default='data/questions.db')
    parser.add_argument('--min_pages', type=int, default=4)
    parser.add_argument("--output_directory", type=str,
                        default="data/wikifier/data/input/",
                        help="Where we write output file")

    flags = parser.parse_args()

    database = QuestionDatabase(flags.database)

    pages = database.questions_with_pages()
    total = 0
    for pp in pages:
        if len(pages[pp]) >= flags.min_pages:
            print(pp, len(pages[pp]))
            for qq in pages[pp]:
                total += 1
                for sentence, word, text in qq.partials():
                    sentence = sentence - 1
                    with open("%s/%i-%i.txt" % (flags.output_directory, qq.qnum, sentence), 'w') as output:
                        output.write("%s\n" % unidecode(text[sentence]))
    print(total)
