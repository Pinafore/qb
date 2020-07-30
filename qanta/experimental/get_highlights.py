import pickle
from tqdm import tqdm
from elasticsearch_dsl import Search
from elasticsearch_dsl.connections import connections

# from qanta.datasets.quiz_bowl import QuestionDatabase

connections.create_connection(hosts=["localhost"])


class color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def get_highlights(text):
    # query top 10 guesses
    s = Search(index="qb_0")[0:10].query(
        "multi_match",
        query=text,
        fields=["wiki_content", "qb_content", "source_content"],
    )
    s = s.highlight("qb_content").highlight("wiki_content")
    results = list(s.execute())
    if len(results) == 0:
        highlights = {"wiki": [""], "qb": [""], "guess": ""}
        return highlights

    guess = results[0]  # take the best answer
    _highlights = guess.meta.highlight

    try:
        wiki_content = list(_highlights.wiki_content)
    except AttributeError:
        wiki_content = [""]

    try:
        qb_content = list(_highlights.qb_content)
    except AttributeError:
        qb_content = [""]

    highlights = {"wiki": wiki_content, "qb": qb_content, "guess": guess.page}
    return highlights


# def test():
#     questions = QuestionDatabase().all_questions()
#     guessdev_questions = [x for x in questions.values() if x.fold == 'guessdev']
#     highlights = get_highlights(questions[0].flatten_text())
#     print(highlights['guess'])
#     for x in highlights['wiki']:
#         print('WIKI|' + x.replace('<em>', color.RED).replace('</em>', color.END))
#     for x in highlights['qb']:
#         print('QUIZ|' + x.replace('<em>', color.RED).replace('</em>', color.END))
#
# def main():
#     questions = QuestionDatabase().all_questions()
#     guessdev_questions = {k: v  for k, v in questions.items()
#             if v.fold == 'guessdev'}
#     highlights = {}
#     for k, v in tqdm(guessdev_questions.items()):
#         highlights[k] = get_highlights(v.flatten_text())
#     with open('guessdev_highlight.pkl', 'wb') as f:
#         pickle.dump(highlights, f)

if __name__ == "__main__":
    test()
