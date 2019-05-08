import sys
import json
import glob
import os
from collections import Counter
import subprocess
import pandas as pd
import click


TOURNAMENT_DEC_15 = 'Adversarial Question Writing UMD December 15'


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def md5sum(filename):
    return subprocess.run(
        f'md5sum {filename}',
        shell=True,
        stdout=subprocess.PIPE,
        check=True
    ).stdout.decode('utf-8').split()[0]


def verify_checksum(checksum, filename):
    if os.path.exists(filename):
        file_checksum = md5sum(filename)
        if checksum != file_checksum:
            raise ValueError(f'Incorrect checksum for: {filename}')
    else:
        raise ValueError(f'File does not exist: {filename}')


@click.group()
def main():
    pass


@main.command()
def validate():
    """
    This command validates that the dataset for trickme paper looks correct.
    It checks that:
    * The number of adversarial questions matches the number in the ID lookup file
    * ID lookup file is a map from qanta_id to source (ir vs rnn, round 1 etc)
    * The number of questions in dataset matches how many the experimental results have
    * That the qanta_id and page for each question match in dataset and experimental results
    """
    with open('data/external/datasets/qanta.trick-no-edits.json') as f:
        questions = [q for q in json.load(f)['questions']]
    with open('data/external/datasets/trickme-id-model.json') as f:
        id_to_model = json.load(f)
        id_to_model = {int(k): v for k, v in id_to_model.items()}

    print('Lengths should be 946')
    print(len(questions))
    print(len(id_to_model))
    print(Counter(id_to_model.values()))
    qid_to_page = {q['qanta_id']: q['page'] for q in questions}
    qids = {q['qanta_id'] for q in questions}
    id_to_model_qids = {k for k in id_to_model.keys()}
    print(len(qids.intersection(id_to_model_qids)))
    df = pd.read_json('output/tacl/all_rounds_df.json')
    # This ID range is used for trickme questions
    df = df[df.qanta_id >= 2_000_000][['qanta_id', 'page']]
    experiments_qid_to_page = {t.qanta_id: t.page for t in df.itertuples()}
    print(len(experiments_qid_to_page))
    for qid, page in experiments_qid_to_page.items():
        if qid_to_page[qid] != page:
            raise ValueError(f'exp: {qid} {page} data: {qid_to_page[qid]}')


@main.command()
def merge():
    """
    Merge various sources of questions:
    - Round 1 data
    - Edited Round 2 rnn data (done by filtering round 2 data out of expo file and using edited file)
    - Round 2 IR data (~100 additional questions)
    We also append information so that in the released data its clear what data comes from what interface.
    Finally, verify checksums and other sanity checks to make sure data is coming from correct place.
    """
    qanta_expo_checksum = 'c56a129b4d9c925187e2e58cc51c0b77'
    trickme_id_model_checksum = 'cb0e26e5c9d1cada7b0b9cd0edb6c9e5'

    verify_checksum(trickme_id_model_checksum, 'data/external/datasets/trickme-id-model.json')
    with open('data/external/datasets/trickme-id-model.json') as f:
        id_to_model = json.load(f)
        id_to_model = {int(k): v for k, v in id_to_model.items()}

    verify_checksum(qanta_expo_checksum, 'data/external/datasets/qanta.trick-no-edits.json')
    with open('data/external/datasets/qanta.trick-no-edits.json') as f:
        data = json.load(f)

    edited_questions = []
    for path in glob.glob('data/external/datasets/qanta.qb-edited-1215.*.json'):
        with open(path) as f:
            edited_questions.extend(json.load(f)['questions'])
    if len(edited_questions) != 410:
        raise ValueError(f'Wrong number of questions: {len(edited_questions)}')

    additional_ir2_checksum = 'e758156eb23f0f307982513be5011268'
    verify_checksum(additional_ir2_checksum, 'data/external/datasets/qanta.trick-additional-ir-round2.json')
    with open('data/external/datasets/qanta.trick-additional-ir-round2.json') as f:
        additional_ir2_questions = json.load(f)['questions']

    questions = data['questions']
    merged_questions = []
    for q in questions:
        q['tournament'] = TOURNAMENT_DEC_15
        q['fold'] = 'adversarial'
        q['trick_id'] = None
        source = id_to_model[q['qanta_id']]
        if source == 'es':
            # Keep IR Round 1
            q['interface'] = 'ir-r1'
            merged_questions.append(q)
        elif source == 'es-2':
            # Add IR Round 2 to additional Round 2
            q['interface'] = 'ir-r2'
            merged_questions.append(q)
        elif source == 'rnn':
            # Replace unedited Round 2 RNN with edited (below)
            q['interface'] = 'rnn'
        else:
            raise ValueError(f'Unrecognized source: {source}')

    # Calculate where to pickup on qanta_ids
    max_id = max(q['qanta_id'] for q in questions)
    curr_qanta_id = max_id + 1
    # These are round 2 RNN questions
    for q in edited_questions:
        q['tournament'] = TOURNAMENT_DEC_15
        q['fold'] = 'adversarial'
        q['interface'] = 'rnn'
        q['trick_id'] = int(q['trick_id'])
        q['qanta_id'] = curr_qanta_id
        curr_qanta_id += 1
        merged_questions.append(q)

    # Append these to existing round 2 IR
    for q in additional_ir2_questions:
        q['tournament'] = TOURNAMENT_DEC_15
        q['fold'] = 'adversarial'
        q['interface'] = 'ir-r2'
        q['trick_id'] = None
        q['qanta_id'] = curr_qanta_id
        curr_qanta_id += 1
        merged_questions.append(q)

    tacl_data = {}
    tacl_data['version'] = data['version']
    tacl_data['maintainer_name'] = data['maintainer_name']
    tacl_data['maintainer_contact'] = data['maintainer_contact']
    tacl_data['maintainer_website'] = data['maintainer_website']
    tacl_data['project_website'] = 'http://trickme.qanta.org'
    tacl_data['bibtex'] = (
        '@inproceedings{Wallace2019Trick,\n'
        '  title={Trick Me If You Can: Human-in-the-loop Generation of Adversarial Question Answering Examples},\n'
        '  author={Eric Wallace and Pedro Rodriguez and Shi Feng and Ikuya Yamada and Jordan Boyd-Graber},\n'
        '  booktitle = "Transactions of the Association for Computational Linguistics"\n'
        '  year={2019},\n'
        '}'
    )
    tacl_data['dependent_checksums'] = {
        'qanta.trick-no-edits.json': qanta_expo_checksum,
        'trickme-id-model.json': trickme_id_model_checksum,
        'qanta.trick-additional-ir-round2.json': additional_ir2_checksum,
    }
    tacl_data['questions'] = merged_questions
    tacl_path = 'data/external/datasets/qanta.tacl-trick.json'
    with open(tacl_path, 'w') as f:
        json.dump(tacl_data, f, indent=2, sort_keys=True)

    print(f'File: {tacl_path} Checksum: {md5sum(tacl_path)}')
    merged_id_to_model = {}
    for q in merged_questions:
        if q['interface'] == 'ir-r2':
            val = 'es-2'
        elif q['interface'] == 'ir-r1':
            val = 'es'
        else:
            val = q['interface']
        merged_id_to_model[q['qanta_id']] = val
    with open('data/external/datasets/merged_trickme-id-model.json', 'w') as f:
        json.dump(merged_id_to_model, f)
    counts = Counter(q['interface'] for q in merged_questions)
    print(f'N: {len(merged_questions)}')
    print(f'{counts}')


if __name__ == '__main__':
    main()
