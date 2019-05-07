import json
import csv
import subprocess
import os
from collections import defaultdict
import click
import yaml
from qanta.ingestion.preprocess import format_qanta_json, add_sentences_
from qanta import qlogging


log = qlogging.get(__name__)

TOURNAMENT_DEC_15 = 'Adversarial Question Writing UMD December 15'


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
def trick_cli():
    pass


@trick_cli.command()
@click.option('--id-model-path', default='data/external/datasets/trickme-id-model.json')
@click.option('--expo-path', default='data/external/datasets/qanta.expo.2018.04.18.json')
@click.option('--version', default='2018.04.18')
@click.argument('rnn-out')
@click.argument('es-out')
def split_ds(id_model_path, expo_path, version, rnn_out, es_out):
    with open(id_model_path) as f:
        lookup = json.load(f)

    with open(expo_path) as f:
        questions = json.load(f)['questions']

    es_questions = []
    rnn_questions = []
    for q in questions:
        qanta_id = str(q['qanta_id'])
        if lookup[qanta_id] == 'es':
            es_questions.append(q)
        elif lookup[qanta_id] == 'rnn':
            rnn_questions.append(q)
        else:
            raise ValueError('Unhandled question source')

    with open(rnn_out, 'w') as f:
        json.dump(format_qanta_json(rnn_questions, version), f)

    with open(es_out, 'w') as f:
        json.dump(format_qanta_json(es_questions, version), f)


@trick_cli.command()
@click.argument('edited-tsv')
@click.argument('out')
def edited_to_json(edited_tsv, out):
    trick_questions = []
    with open(edited_tsv) as f:
        # Burn the header
        next(f)
        for line in f:
            fields = line.split('\t')
            trick_id = fields[0]
            question = fields[1]
            answer = fields[2]
            round_ = fields[6]
            trick_questions.append({'trick_id': trick_id, 'question': question, 'answer': answer, 'round': round_})

    with open(out, 'w') as f:
        json.dump(trick_questions, f)


@trick_cli.command()
def format_additional():
    """
    Additional questions were added to dataset, this processes the csv version to match
    the dataset format while verifying that page info is valid.
    """
    titles_checksum = '6fa134836b3a7e3b562cdaa8ad353f2d'
    verify_checksum(titles_checksum, 'data/external/wikipedia/wikipedia-titles.2018.04.18.json')
    with open('data/external/wikipedia/wikipedia-titles.2018.04.18.json') as f:
        titles = set(json.load(f))

    trick_checksum = '905594aab776ddb10b0d7f36d30633a2'
    verify_checksum(trick_checksum, 'data/external/datasets/trick-additional.csv')

    with open('data/external/datasets/trick-additional.csv') as f:
        # Ignore header row
        rows = list(csv.reader(f))[1:]

    questions = []
    for _, text, page in rows:
        page = page.replace(' ', '_')
        if page not in titles:
            log.info(f'Page not in titles: {page}')
        questions.append({
            'text': text,
            'answer': page,
            'page': page,
            'fold': 'advtest',
            'year': 2018,
            'dataset': 'trickme',
            'proto_id': None,
            'qdb_id': None,
            'difficulty': None,
            'category': None,
            'subcategory': None,
            'qanta_id': None,
            'tournament': TOURNAMENT_DEC_15,
            'gameplay': False,
            'interface': 'ir-r2',
            'dependent_checksums': {
                'trick-additional.csv': trick_checksum,
                'wikipedia-titles.2018.04.18.json': titles_checksum
            }
        })
    add_sentences_(questions, parallel=False)
    dataset = format_qanta_json(questions, '2018.04.18')
    path_formatted = 'data/external/datasets/qanta.trick-additional-ir-round2.json'
    with open(path_formatted, 'w') as f:
        json.dump(dataset, f)
    log.info(f'File: {path_formatted} Checksum: {md5sum(path_formatted)}')


@trick_cli.command()
@click.option('--answer-map-path', default='data/internal/trickme_answer_map.yml')
@click.option('--qanta-ds-path', default='data/external/datasets/qanta.mapped.2018.04.18.json')
@click.option('--wiki-titles-path', default='data/external/wikipedia/wikipedia-titles.2018.04.18.json')
@click.option('--trick-path', default='data/external/datasets/trickme-expo.json')
@click.option('--id-model-path', default='data/external/datasets/trickme-id-model.json')
@click.option('--out-path', default='data/external/datasets/qanta.expo.2018.04.18.json')
@click.option('--start-idx', default=2000000)
@click.option('--version', default='2018.04.18')
@click.option('--fold', default='advtest')
@click.option('--year', default=2018)
@click.option('--tournament', default=TOURNAMENT_DEC_15)
@click.option('--separate-rounds', default=False, is_flag=True)
def trick_to_ds(answer_map_path, qanta_ds_path, wiki_titles_path, trick_path,
                id_model_path, out_path,
                start_idx, version, fold, year, tournament,
                separate_rounds):
    with open(answer_map_path) as f:
        answer_map = yaml.load(f)

    with open(qanta_ds_path) as f:
        qanta_ds = json.load(f)['questions']
    answer_set = {q['page'] for q in qanta_ds if q['page'] is not None}
    with open(wiki_titles_path) as f:
        titles = set(json.load(f))
    lookup = {a.lower().replace(' ', '_'): a for a in answer_set}
    id_model_map = {}
    skipped = 0
    with open(trick_path) as f:
        questions = []
        for i, q in enumerate(json.load(f)):
            if 'Question' in q:
                text = q['Question']
            elif 'question' in q:
                text = q['question']
            else:
                raise ValueError('Could not find question field in question')

            if 'Answer' in q:
                answer = q['Answer'].replace(' ', '_')
            elif 'answer' in q:
                answer = q['answer'].replace(' ', '_')
            else:
                raise ValueError('Could not find answer field in question')

            if 'trick_id' in q:
                trick_id = q['trick_id']
            else:
                trick_id = None

            if len(answer) == 0:
                raise ValueError(f'Empty answer for trick_id={trick_id}')
            elif len(text) == 0:
                raise ValueError(f'Empty text for trick_id={trick_id}')

            if answer in titles or answer in answer_set:
                page = answer
            elif answer in lookup:
                page = lookup[answer]
            elif answer in answer_map:
                m_page = answer_map[answer]
                if m_page is None:
                    if 'model' in q:
                        log.info(f'Explicitly Skipping {answer}, int-model: {q["model"]}')
                    else:
                        log.info(f'Explicitly Skipping {answer}')
                    continue  # Skip this explicitly
                elif m_page in answer_set:
                    page = m_page
                else:
                    raise ValueError(f'{m_page} not in answer set\n Q: {text}')
            else:
                log.error(f'Unhandled Skipping: idx: {i} trick_id: {trick_id} A: "{answer}"\nQ:"{text}"')
                skipped += 1
                continue

            q_out = {
                'text': text,
                'answer': answer,
                'page': page,
                'fold': fold,
                'year': year,
                'dataset': 'trickme',
                'proto_id': None,
                'qdb_id': None,
                'difficulty': None,
                'category': None,
                'subcategory': None,
                'qanta_id': start_idx + i,
                'tournament': tournament,
                'gameplay': False,
                'trick_id': trick_id
            }
            if 'email' in q:
                q_out['author_email'] = q['email']
            if 'category' in q and q['category'] != "None":
                q_out['category'] = q['category']
            if 'round' in q:
                q_out['round'] = q['round']
            if 'model' in q:
                id_model_map[q_out['qanta_id']] = q['model']
            questions.append(q_out)
        log.info(f'Total: {len(questions)} Skipped: {skipped}')
        add_sentences_(questions, parallel=False)
        if separate_rounds:
            rounds = defaultdict(list)
            for q in questions:
                rounds[q['round']].append(q)
            for name, round_questions in rounds.items():
                dataset = format_qanta_json(round_questions, version)
                file_name = out_path.split('.')
                if file_name[-1] == 'json':
                    file_name.pop()
                    file_name.extend([name, 'json'])
                else:
                    file_name.extend([name, 'json'])
                round_out_path = '.'.join(file_name)
                log.info(f'Writing round {name} to {round_out_path}')
                with open(round_out_path, 'w') as f:
                    json.dump(dataset, f)
        else:
            dataset = format_qanta_json(questions, version)

            with open(out_path, 'w') as f:
                json.dump(dataset, f)

        with open(id_model_path, 'w') as f:
            json.dump(id_model_map, f)


@trick_cli.command()
@click.argument('es_file')
@click.argument('rnn_file')
@click.argument('out_file')
def merge(es_file, rnn_file, out_file):
    out = []
    with open(es_file) as f:
        for q in json.load(f):
            q['model'] = 'es'
            out.append(q)

    with open(rnn_file) as f:
        for q in json.load(f):
            q['model'] = 'rnn'
            out.append(q)

    with open(out_file, 'w') as f:
        json.dump(out, f)


@trick_cli.command()
@click.argument('expo_in')
@click.argument('server_out')
def to_server(expo_in, server_out):
    with open(expo_in) as f:
        questions = json.load(f)['questions']
        server_questions = []
        for q in questions:
            text = q['text']
            answer = q['page']
            qid = q['qanta_id']
            fold = 'test'
            server_questions.append({
                'question': text,
                'answer': answer,
                'qid': qid,
                'fold': fold
            })

    with open(server_out, 'w') as f:
        json.dump({'questions': server_questions}, f)
