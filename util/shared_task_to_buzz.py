import nltk
import argparse
from csv import DictWriter, DictReader
from random import shuffle
from string import split

kPOWER = "(*)"
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

def find_powers(questions):
    powers = {}
    for ii in questions:
        for ss in questions[ii]:
            if kPOWER in ss:
                words = ss.split()
                powers[ii] = words[words.index(kPOWER) + 1]
    return powers

def word_position_to_sent(questions, question, position):
    assert question in questions, "%i not in questions" % question
    count = 0
    for ss, sent in enumerate(questions[question]):
        for ww, word in enumerate(sent.split()):
            count += 1
            if count >= position:
                return ss, ww
    return ss, ww

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--user', type=int, default=-1,
                        help='The user selected')
    parser.add_argument('--shared_task', type=str, default='',
                        help="Answers from shared task")
    parser.add_argument('--shared_task_offset', type=int, default=0,
                        help="Starting ID for shared task input")
    parser.add_argument('--question_offset', type=int, default=0,
                        help="Starting ID for questions file offset")
    parser.add_argument('--questions', type=str, default='',
                        help="Text of questions")
    parser.add_argument('--sent_sep_ques', type=str,
                        default='results/st/questions.csv',
                        help="Sentence separated guesses")
    parser.add_argument('--final', type=str,
                        default='results/st/final.csv',
                        help="Where we write final answers")
    parser.add_argument('--power', type=str,
                        default='results/st/power.csv',
                        help="Where we write power marks")
    parser.add_argument('--buzz', type=str,
                        default='results/st/buzz.csv',
                        help='Where we write resulting buzzes')
    
    flags = parser.parse_args()

    results = {}

    # Read in the questions so that we can convert absolute word
    # positions to sentence, word positions
    questions = {}
    answers = {}
    for ii in DictReader(open(flags.questions)):
        question_id = int(ii["id"]) + flags.question_offset
        questions[question_id] = sent_detector.tokenize(ii['text'])
        answers[question_id] = ii['answer']
    
    for ii in open(flags.shared_task):
        user, question, pos, guess = split(ii.strip(), maxsplit=3)
        user = int(user)
        question = int(question) + flags.shared_task_offset
        pos = int(pos)

        if flags.user > 0 and flags.user == user:
            results[question] = (pos, guess)
        else:
            if question not in results or results[question][0] > pos:
                results[question] = (pos, guess)

    powers = find_powers(questions)
                
    # Write out the questions, buzzes, and finals
    o_questions = DictWriter(open(flags.sent_sep_ques, 'w'),
                             ['id', 'answer', 'sent', 'text'])
    o_questions.writeheader()
    
    o_buzz = DictWriter(open(flags.buzz, 'w'),
                        ['question', 'sentence', 'word', 'page', 'evidence',
                         'final', 'weight'])
    o_buzz.writeheader()
    
    o_final = DictWriter(open(flags.final, 'w'), ['question', 'answer'])
    o_final.writeheader()
    
    o_power = DictWriter(open(flags.power, 'w'), ['question', 'word'])
    o_power.writeheader()

    for question in results:
        pos, guess = results[question]
        ss, tt = word_position_to_sent(questions, question, pos)

        for sent_offset, sent in enumerate(questions[question]):
            question_line = {}
            question_line['id'] = question
            question_line['answer'] = answers[question]
            question_line['sent'] = sent_offset
            question_line['text'] = sent
            o_questions.writerow(question_line)

        buzz_line = {}
        buzz_line['question'] = question
        buzz_line['sentence'] = ss
        buzz_line['word'] = tt
        buzz_line['page'] = guess
        buzz_line['final'] = 1
        buzz_line['weight'] = 1.0
        o_buzz.writerow(buzz_line)

        final_line = {}
        final_line['question'] = question
        final_line['answer'] = guess
        o_final.writerow(final_line)

        power_line = {}
        power_line['question'] = question
        power_line['word'] = powers[question].strip()
        o_power.writerow(power_line)
        
