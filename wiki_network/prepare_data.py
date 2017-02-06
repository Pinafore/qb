import pickle
import click
import re
from nltk.corpus import stopwords
from qanta.datasets.quiz_bowl import QuestionDatabase


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
            if len(page) > 2 and re.match(r"^[a-zA-Z0-9_\(\)']+$", page) and page not in stop_words:
                f.write(line.strip().lower())
            else:
                f.write('@')
            f.write('\n')
    titles_file.close()


if __name__ == '__main__':
    main()

