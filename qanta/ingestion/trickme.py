import json
import click
import yaml
from qanta.ingestion.preprocess import format_qanta_json, add_sentences_
from qanta import qlogging


log = qlogging.get(__name__)


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
@click.option('--answer-map-path', default='data/internal/trickme_answer_map.yml')
@click.option('--qanta-ds-path', default='data/external/datasets/qanta.mapped.2018.04.18.json')
@click.option('--trick-path', default='data/external/datasets/trickme-expo.json')
@click.option('--id-model-path', default='data/external/datasets/trickme-id-model.json')
@click.option('--out-path', default='data/external/datasets/qanta.expo.2018.04.18.json')
@click.option('--start-idx', default=2000000)
@click.option('--version', default='2018.04.18')
@click.option('--fold', default='advtest')
@click.option('--year', default=2018)
@click.option('--tournament', default='Adversarial Question Writing UMD December 15')
def trick_to_ds(answer_map_path, qanta_ds_path, trick_path, id_model_path, out_path,
                start_idx, version, fold, year, tournament):
    with open(answer_map_path) as f:
        answer_map = yaml.load(f)

    with open(qanta_ds_path) as f:
        qanta_ds = json.load(f)['questions']
    answer_set = {q['page'] for q in qanta_ds if q['page'] is not None}
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

            if len(answer) == 0 or len(text) == 0:
                raise ValueError('Empty answer or text')

            if answer in answer_set:
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
                log.error(f'Unhandled Skipping: idx: {i} A: "{answer}"\nQ:"{text}"')
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
                'gameplay': False
            }
            if 'email' in q:
                q_out['author_email'] = q['email']
            if 'category' in q and q['category'] != "None":
                q_out['category'] = q['category']
            if 'model' in q:
                id_model_map[q_out['qanta_id']] = q['model']
            questions.append(q_out)
        log.info(f'Total: {len(questions)} Skipped: {skipped}')
        add_sentences_(questions, parallel=False)
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
