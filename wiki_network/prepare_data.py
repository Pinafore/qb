import pickle
import click
import re
from nltk.corpus import stopwords
from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.preprocess import format_guess


@click.group()
def main():
    pass


@main.command()
def generate_questions():
    with open('data/100_possible_questions.pickle', 'rb') as f:
        qs = pickle.load(f)

    with open('data/qb_questions.txt', 'w') as f:
        for q in qs:
            f.write(q.flatten_text())
            f.write('\n')

    db = QuestionDatabase()
    answers = db.all_answers().values()
    with open('data/answers.txt', 'w') as f:
        for a in answers:
            f.write(a.lower().replace(' ', '_'))
            f.write('\n')


@main.command()
def preprocess_titles():
    stop_words = set(stopwords.words('english'))
    titles_file = open('data/titles-sorted.txt')
    with open('data/processed-titles-sorted.txt', 'w') as f:
        for line in titles_file:
            page = line.strip().lower()
            if len(page) > 2 and re.match(r"^[a-zA-Z0-9_()']+$", page) and page not in stop_words:
                f.write(line.strip().lower())
            else:
                f.write('@')
            f.write('\n')
    titles_file.close()


@main.command()
def n2v_edge_list():
    # Indexing on the links/titles starts at 1 in the data
    i = 1
    n_invalid_titles = 0
    n_valid_titles = 0
    ind_to_title = {}
    title_to_ind = {}
    with open('data/processed-titles-sorted.txt') as f:
        for line in f:
            line = line.strip()
            if line == '@':
                n_invalid_titles += 1
            else:
                normed_title = format_guess(line)
                ind_to_title[i] = normed_title
                title_to_ind[normed_title] = i
                n_valid_titles += 1
            i += 1
    output = open('data/edge_list.txt', 'w')
    with open('data/links-simple-sorted.txt') as f:
        for line in f:
            u, v_list = line.split(':')
            u = int(u)
            if u in ind_to_title:
                v_list = [int(v) for v in v_list.split()]
                for v in v_list:
                    if v in ind_to_title:
                        if u < v:
                            output.write('{} {}\n'.format(u, v))
                        elif u > v:
                            output.write('{} {}\n'.format(v, u))
    output.close()

if __name__ == '__main__':
    main()

