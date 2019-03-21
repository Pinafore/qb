"""
The purpose of the code here is to provide a way to easily map existing data using the
same process the qanta dataset uses
"""
import json
import click
import yaml
from qanta.util.io import safe_open
from qanta.ingestion.answer_mapping import create_answer_map, write_answer_map, unmapped_to_mapped_questions
from qanta.ingestion.annotated_mapping import PageAssigner
from qanta.ingestion.preprocess import add_sentences_, format_qanta_json
from qanta.ingestion.pipeline import QANTA_PREPROCESSED_DATASET_PATH, DS_VERSION


@click.command(name='map')
@click.option('--start-idx', default=3_000_000, type=int, help="Starting qanta_id index")
def ingestion_cli(start_idx):
    """
    Input format is for jason's HS project, but can be changed. The original code for answer
    mapping was designed to map everything over multiple passes, not yield a callable function to map
    an arbitrary answer line to a QB answer. Rather than implement this, a hacky way to achieve similar
    functionality to map a new dataset is to combine already mapped questions with new questions, have
    the code map answer for both at the same time, then only use the mappings from the new questions.
    There are some edge cases, but this should in general work (hopefully).
    """
    with open(QANTA_PREPROCESSED_DATASET_PATH) as f:
        unmapped_questions = json.load(f)['questions']

    with open('data/external/high_school_project/quizdb-20190313164802.json') as f:
        raw_questions = json.load(f)['data']['tossups']

    new_questions = []
    idx = start_idx
    for q in raw_questions:
        new_questions.append({
            'qanta_id': idx,
            'text': q['text'],
            'answer': q['answer'],
            'page': None,
            'category': None,
            'subcategory': None,
            'tournament': q['tournament']['name'],
            'difficulty': q['tournament']['difficulty'],
            'year': int(q['tournament']['year']),
            'proto_id': None,
            'qdb_id': q['id'],
            'dataset': 'quizdb.org',
            'fold': 'guesstest'
        })
        idx += 1
    questions = unmapped_questions + new_questions
    answer_map, amb_answer_map, unbound_answers, report = create_answer_map(questions)
    with safe_open('data/external/high_school_project/automatic_report.json', 'w') as f:
        json.dump(report, f)

    write_answer_map(
        answer_map, amb_answer_map, unbound_answers,
        'data/external/high_school_project/answer_map.json',
        'data/external/high_school_project/unbound_answers.json'
    )
    with open('data/internal/page_assignment/unmappable.yaml') as f:
        unmappable = yaml.load(f)

    page_assigner = PageAssigner()
    mapping_report = unmapped_to_mapped_questions(
        new_questions,
        answer_map, amb_answer_map,
        unmappable, page_assigner
    )

    add_sentences_(new_questions)
    with open('data/external/high_school_project/qanta.acf-regionals-2018.json', 'w') as f:
        json.dump(format_qanta_json(new_questions, DS_VERSION), f)

    with open('data/external/high_school_project/mapping_report.json', 'w') as f:
        json.dump(mapping_report, f)
