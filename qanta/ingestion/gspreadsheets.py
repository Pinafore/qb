from typing import Dict
import json
import os
import pandas as pd

from qanta.util.constants import QANTA_MAP_REPORT_PATH, GUESSER_TRAIN_FOLD, BUZZER_TRAIN_FOLD
from qanta.datasets.quiz_bowl import QantaDatabase, Question


UNMAPPED_COLUMNS = [
    'result', 'reason',
    'proto_id', 'qdb_id', 'qanta_id',
    'last_sentence',
    'answer', 'page'
]

DISAGREE_COLUMNS = [
    'result', 'reason',
    'proto_id', 'qdb_id', 'qanta_id', 'is_train',
    'last_sentence',
    'answer', 'automatic_page', 'annotated_page'
]


def last_sentence(q_dict):
    text = q_dict['text']
    tokenizations = q_dict['tokenizations']
    start, end = tokenizations[-1]
    return text[start:end]


def unmapped_rows(match_report, questions):
    rows = []
    for q in questions:
        qanta_id = str(q['qanta_id'])
        result = match_report[qanta_id]
        reason = ''
        if result['annotated_error'] is not None:
            reason += result['annotated_error']
        if result['automatic_error'] is not None:
            reason += result['automatic_error']
        rows.append((
            result['result'], reason,
            q['proto_id'], q['qdb_id'], q['qanta_id'],
            last_sentence(q), q['answer'], None
        ))
    return rows


def create_answer_mapping_csvs(output_dir='data/external/answer_mapping'):
    with open(QANTA_MAP_REPORT_PATH) as f:
        report = json.load(f)
        match_report = report['match_report']
    db = QantaDatabase()
    qb_lookup: Dict[int, Question] = {q.qanta_id: q for q in db.all_questions}
    train_rows = unmapped_rows(match_report, report['train_unmatched'])
    test_rows = unmapped_rows(match_report, report['test_unmatched'])
    train_df = pd.DataFrame.from_records(train_rows, columns=UNMAPPED_COLUMNS)
    test_df = pd.DataFrame.from_records(test_rows, columns=UNMAPPED_COLUMNS)
    train_df.to_csv(os.path.join(output_dir, 'unmapped_train.csv'))
    test_df.to_csv(os.path.join(output_dir, 'unmapped_test.csv'))

    disagree_rows = []
    for qanta_id, row in match_report.items():
        if row['result'] == 'disagree':
            q = qb_lookup[int(qanta_id)]
            start, end = q.tokenizations[-1]
            is_train = q.fold == GUESSER_TRAIN_FOLD or q.fold == BUZZER_TRAIN_FOLD
            disagree_rows.append((
                'disagree', None,
                q.proto_id, q.qdb_id, q.qanta_id, is_train, q.text[start:end],
                q.answer, row['automatic_page'], row['annotated_page']
            ))
    disagree_df = pd.DataFrame.from_records(disagree_rows, columns=DISAGREE_COLUMNS)
    disagree_df[disagree_df.is_train == True].to_csv(os.path.join(output_dir, 'disagree_train.csv'))
    disagree_df[disagree_df.is_train == False].to_csv(os.path.join(output_dir, 'disagree_test.csv'))
