
from collections import defaultdict

import pickle
import time

from page_assignment.active_learning_for_matching import ActiveLearner
from util.qdb import QuestionDatabase


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="apply wikipedia pages")
    parser.add_argument("--db", default='data/questions.db', type=str,
                        help="The question database")
    parser.add_argument("--match_location", type=str,
                        default='data/map/ans_to_wiki_',
                        help="Where we read matches learned")

    flags = parser.parse_args()

    start = time.time()
    print("Loading db..")
    db = QuestionDatabase(flags.db)
    print("Loading classifier...")
    classifier = ActiveLearner(None, flags.match_location, [])

    for question, page in classifier.human_labeled():
        ans_type = ""
        db.set_answer_page(question, page, ans_type)
        print(question, page, "GIVEN", ans_type)
