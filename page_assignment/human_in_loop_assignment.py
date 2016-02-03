import sys
print(sys.path)
from util.qdb import QuestionDatabase
import argparse
from page_assignment.active_learning_for_matching import ActiveLearner, simple_menu
from page_assignment.client import TitleFinder, normalize
import operator

kBAD_ANSWERS = ["", "red river", "the", "figaro", "normal", "s", "p"]

if __name__ == "__main__":
    args = argparse.ArgumentParser('Interactive assign pages to questions')
    args.add_argument('--database', type=str, default='data/questions.db',
                      help='sqlite3 database of questions')
    args.add_argument('--titles', type=str, default='data/wiki_index.pkl',
                      help='page title candiates')
    args.add_argument('--labels', type=str, default='data/map/ans_to_wiki',
                      help='write page assignment answers')
    args = args.parse_args()

    # Open up the database
    d = QuestionDatabase(args.database)
    page_diversity = d.answer_map(normalize)

    # Set up the active learner for writing assignments
    al = ActiveLearner(None, args.labels)
    existing_labels = set(x[0] for x in al.human_labeled())

    # get the candidates we want to assign to pages
    answers = d.unmatched_answers(existing_labels)
    print(answers.keys()[:10])

    # Open up the title finder
    tf = TitleFinder(open(args.titles))

    for ans, count in sorted(answers.items(), key=lambda x: sum(x[1].values()),
                             reverse=True):
        if ans in kBAD_ANSWERS:
            continue
        choices = list(tf.query(ans))
        print("--------- (%i)" % sum(count.values()))
        print(ans)

        normalized_answer = normalize(ans)
        if sum(page_diversity[normalized_answer].values()) >= 5 and \
                len(page_diversity[normalized_answer]) == 1:
            page = page_diversity[normalized_answer].keys()[0]
            print("Autoassigning %s to %s" % (ans, page))
        else:
            page = simple_menu([x[0] for x in choices], tf._index,
                               [x[1] for x in choices])
        if page:
            for query in count:
                for question in d.questions_by_answer(query):
                    if not question.page:
                        al.remember(question.qnum, page)
                        print("%i\t%s" % (question.qnum,
                                          question.text[max(question.text)]))
        else:
            for query in count:
                for question in d.questions_by_answer(query):
                    if not question.page:
                        print("%i\t%s" % (question.qnum,
                                          question.text[max(question.text)]))

        al.dump(args.labels)
