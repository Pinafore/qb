from tqdm import tqdm
from elasticsearch_dsl.connections import connections
from elasticsearch_dsl import DocType, Text, Keyword, Search, Index
from qanta.util.constants import GUESSER_DEV_FOLD
from qanta.guesser.abstract import AbstractGuesser
from qanta.datasets.quiz_bowl import QuizBowlDataset
from qanta.guesser.experimental.elasticsearch_instance_of import (
    ElasticSearchWikidataGuesser,
)
from qanta.guesser.experimental.elasticsearch_instance_of import ElasticSearchIndex

INDEX_NAME = "qb_ir_instance_of"

gspec = AbstractGuesser.list_enabled_guessers()[0]
guesser_dir = AbstractGuesser.output_path(gspec.guesser_module, gspec.guesser_class, "")
guesser = ElasticSearchWikidataGuesser.load(guesser_dir)
es_index = ElasticSearchIndex()


def recursive_guess(question, k=0):
    p_class, p_prob = guesser.test_instance_of([question])[0]
    first_guesses = search_not(question, p_class)
    print("First round")
    for x in first_guesses:
        print(x)
    print()

    print("Second round")
    new_guesses = []
    for i in range(k):
        guess = first_guesses[i][0]
        question += " " + " ".join(guess.split("_"))
        guesses = es_index.search(question, p_class, p_prob, 0.6)
        for x in guesses:
            print(x)
        print()
        new_guesses.append(guesses[0])

    new_guesses = sorted(new_guesses, key=lambda x: x[1])[::-1]
    return new_guesses[0][0]


def search_not(text, p_class):
    query_length = len(text.split())
    s = Search(index=INDEX_NAME).query(
        "multi_match", query=text, fields=["wiki_content", "qb_content"]
    )
    results = s.execute()
    results = [x for x in results if x.instance_of != p_class]
    return [(r.page, r.meta.score / query_length) for r in results]


def test():
    dataset = QuizBowlDataset(guesser_train=True)
    questions = dataset.questions_by_fold([GUESSER_DEV_FOLD])
    questions = questions[GUESSER_DEV_FOLD]

    i = 10
    question = questions[i]
    guess = recursive_guess(question.text[0], k=1)
    print(question.page)
    print(question.text[0])


def main():
    dataset = QuizBowlDataset(guesser_train=True)
    questions = dataset.questions_by_fold([GUESSER_DEV_FOLD])
    questions = questions[GUESSER_DEV_FOLD]
    correct = 0
    for question in tqdm(questions):
        guess = recursive_guess(question.text[0], 3)
        correct += guess == question.page
    print(correct / len(questions))


if __name__ == "__main__":
    test()
