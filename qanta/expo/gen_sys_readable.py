import argparse
from math import nan
import pandas as pd
from tqdm import tqdm


def create_parser():
    parser = argparse.ArgumentParser(description="")
    # 2024_470/start.csv
    parser.add_argument(
        "--question_file",
        type=str
    )
    # GAMEPLAY 10:09PM on September 25 2024 (model).csv
    parser.add_argument(
        "--model_outfile",
        type=str
    )
    return parser.parse_args()


def main(args):
    gameplay_log = pd.read_csv(args.model_outfile).to_dict(orient='records')
    questions_data = pd.read_csv(args.question_file).to_dict(orient='records')
    
    qid2runs = {}
    for q_run in questions_data:
        qid = q_run["id"]
        if qid not in qid2runs:
            qid2runs[qid] = {
                'runs': [],
                'answer': q_run['answer'],
                'model_guess': None,
                'model_correctness': None,
            }
        # TODO: We might be able to find a better way to handle this
        qid2runs[qid]['runs'].append(q_run['text'].replace('\n', ' '))
    for qid in qid2runs:
        qid2runs[qid]['question'] = ' '.join(qid2runs[qid]['runs'])

    for record in tqdm(gameplay_log):
        qid = record['qid']
        # TODO: We might be able to find a better way to handle this
        ori_sent = record['sentence'].replace('\n', ' ')
        updated_sent = record['sentence'].replace('\n', ' ').replace(record['model_buzz'], '%s (#)' % record['model_buzz'])

        question = qid2runs[qid]['question']
        question = question.replace(ori_sent, updated_sent)

        qid2runs[qid]['question'] = question
        qid2runs[qid]['model_guess'] = record['model_guess']
        qid2runs[qid]['model_correctness'] = record['model_correctness']
    
    with open(args.model_outfile.replace('.csv', '_sys_readable.txt'), 'w') as f:
        for qid in qid2runs:
            f.write("Question: %s\n" % qid)
            f.write(qid2runs[qid]['question'] + '\n\n')
            f.write("Model guess: %s\n" % qid2runs[qid]['model_guess'])
            f.write("Model correctness: %s\n" % qid2runs[qid]['model_correctness'])
            f.write("Answer: %s\n" % qid2runs[qid]['answer'])
            f.write('\n')


if __name__ == "__main__":
    args = create_parser()
    main(args)
