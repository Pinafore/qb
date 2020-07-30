import os
import pickle
from tqdm import tqdm
import wikipedia
from requests.exceptions import ConnectionError
from wikipedia.exceptions import DisambiguationError, PageError

from qanta.util.environment import BONUS_ANSWER_PAGES
from qanta.util.multiprocess import _multiprocess
from qanta.datasets.quiz_bowl import BonusQuestionDatabase


def download_pages():
    bonus_questions = BonusQuestionDatabase().all_questions()
    train_answers = set()
    for q in bonus_questions.values():
        train_answers.update(q.pages)

    if os.path.isfile(BONUS_ANSWER_PAGES):
        with open(BONUS_ANSWER_PAGES, "rb") as f:
            try:
                pages = pickle.load(f)
                print("loaded {} pages".format(len(pages)))
            except EOFError:
                pages = dict()
    else:
        pages = dict()

    train_answers = train_answers - set(pages.keys())

    for answer in tqdm(train_answers):
        if answer in pages:
            continue
        try:
            page = wikipedia.page(answer)
        except (DisambiguationError, PageError, ConnectionError) as e:
            if isinstance(e, DisambiguationError):
                pages[answer] = None
                continue
            if isinstance(e, PageError):
                pages[answer] = None
                continue
            if isinstance(e, ConnectionError):
                break
        try:
            pages[answer] = [
                page.title,
                page.content,
                page.links,
                page.summary,
                page.categories,
                page.url,
                page.pageid,
            ]
        except ConnectionError:
            break

    with open(BONUS_ANSWER_PAGES, "wb") as f:
        pickle.dump(pages, f)


if __name__ == "__main__":
    download_pages()
