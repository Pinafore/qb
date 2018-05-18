#!/usr/bin/env python
"""
CLI utilities for QANTA
"""

import sqlite3
import yaml
import csv
from collections import defaultdict
import json
from os import path
import click
from typing import Dict, Optional
from jinja2 import Environment, PackageLoader
import tqdm

from qanta import qlogging
from qanta.guesser.abstract import AbstractGuesser
from qanta.guesser.elasticsearch import create_es_config, start_elasticsearch, stop_elasticsearch
from qanta.util.environment import ENVIRONMENT
from qanta.util.io import safe_open, shell, get_tmp_filename
from qanta.util.constants import QANTA_SQL_DATASET_PATH
from qanta.hyperparam import expand_config

log = qlogging.get('cli')

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    log.info("QANTA starting with configuration:")
    for k, v in ENVIRONMENT.items():
        log.info("{0}={1}".format(k, v))


@main.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=5000)
@click.option('--debug', default=False)
@click.argument('guessers', nargs=-1)
def guesser_api(host, port, debug, guessers):
    if debug:
        log.warn(
            'WARNING: debug mode in flask can expose environment variables, including AWS keys, NEVER use this when the API is exposed to the web')
        log.warn('Confirm that you would like to enable flask debugging')
        confirmation = input('yes/no:\n').strip()
        if confirmation != 'yes':
            raise ValueError('Most confirm enabling debug mode')

    AbstractGuesser.multi_guesser_web_api(guessers, host=host, port=port, debug=debug)


def run_guesser(n_times, workers, guesser_qualified_class):
    for _ in range(n_times):
        if 'qanta.guesser' not in guesser_qualified_class:
            log.error('qanta.guesser not found in guesser_qualified_class, this is likely an error, exiting.')
            return
        shell('rm -rf /tmp/qanta')
        shell(f'rm -rf output/guesser/{guesser_qualified_class}')
        shell(f'luigi --local-scheduler --module qanta.pipeline.guesser --workers {workers} AllSingleGuesserReports')


@main.command()
@click.option('--n_times', default=1)
@click.option('--workers', default=1)
@click.argument('guesser_qualified_class')
def guesser_pipeline(n_times, workers, guesser_qualified_class):
    run_guesser(n_times, workers, guesser_qualified_class)


@main.command()
@click.option('--n', default=20)
def sample_answer_pages(n):
    """
    Take a random sample of n questions, then return their answers and pages
    formatted for latex in the journal paper
    """
    conn = sqlite3.connect(QANTA_SQL_DATASET_PATH)
    c = conn.cursor()
    rows = c.execute(f'select answer, page from questions order by random() limit {n}')
    latex_format = r'{answer} & {page}\\ \hline'
    for answer, page in rows:
        answer = answer.replace('{', r'\{').replace('}', r'\}').replace('_', r'\_')
        if page == '':
            page = r'\textbf{No Mapping Found}'
        else:
            page = page.replace('{', r'\{').replace('}', r'\}').replace('_', r'\_')
        print(latex_format.format(answer=answer, page=page))


@main.command()
@click.argument('base_file')
@click.argument('hyper_file')
@click.argument('output_file')
def hyper_to_conf(base_file, hyper_file, output_file):
    expand_config(base_file, hyper_file, output_file)


def get_slurm_config_value(name: str, default_config: Dict, guesser_config: Optional[Dict]):
    if guesser_config is None:
        return default_config[name]
    else:
        if name in guesser_config:
            return guesser_config[name]
        else:
            return default_config[name]


@main.command()
@click.option('--slurm-config-file', default='slurm-config.yaml')
@click.option('--task', default='GuesserPerformance')
@click.argument('output_dir')
def generate_guesser_slurm(slurm_config_file, task, output_dir):
    with open(slurm_config_file) as f:
        slurm_config = yaml.load(f)
        default_slurm_config = slurm_config['default']
    env = Environment(loader=PackageLoader('qanta', 'slurm/templates'))
    template = env.get_template('guesser-luigi-template.sh')
    enabled_guessers = list(AbstractGuesser.list_enabled_guessers())

    for i, gs in enumerate(enabled_guessers):
        if gs.guesser_class == 'ElasticSearchGuesser':
            raise ValueError('ElasticSearchGuesser is not compatible with slurm')
        elif gs.guesser_class in slurm_config:
            guesser_slurm_config = slurm_config[gs.guesser_class]
        else:
            guesser_slurm_config = None
        partition = get_slurm_config_value('partition', default_slurm_config, guesser_slurm_config)
        qos = get_slurm_config_value('qos', default_slurm_config, guesser_slurm_config)
        mem_per_cpu = get_slurm_config_value('mem_per_cpu', default_slurm_config, guesser_slurm_config)
        gres = get_slurm_config_value('gres', default_slurm_config, guesser_slurm_config)
        max_time = get_slurm_config_value('max_time', default_slurm_config, guesser_slurm_config)
        cpus_per_task = get_slurm_config_value('cpus_per_task', default_slurm_config, guesser_slurm_config)
        script = template.render({
            'task': task,
            'guesser_module': gs.guesser_module,
            'guesser_class': gs.guesser_class,
            'dependency_module': gs.dependency_module,
            'dependency_class': gs.dependency_class,
            'config_num': gs.config_num,
            'partition': partition,
            'qos': qos,
            'mem_per_cpu': mem_per_cpu,
            'max_time': max_time,
            'gres': gres,
            'cpus_per_task': cpus_per_task
        })
        slurm_file = path.join(output_dir, f'slurm-{i}.sh')
        with safe_open(slurm_file, 'w') as f:
            f.write(script)

    singleton_path = 'qanta/slurm/templates/guesser-singleton.sh'
    singleton_output = path.join(output_dir, 'guesser-singleton.sh')
    shell(f'cp {singleton_path} {singleton_output}')

    master_template = env.get_template('guesser-master-template.sh')
    master_script = master_template.render({
        'script_list': [
                           path.join(output_dir, f'slurm-{i}.sh') for i in range(len(enabled_guessers))
                       ] + [singleton_output]
    })
    with safe_open(path.join(output_dir, 'slurm-master.sh'), 'w') as f:
        f.write(master_script)


@main.command()
@click.option('--partition', default='dpart')
@click.option('--qos', default='batch')
@click.option('--mem-per-cpu', default='8g')
@click.option('--max-time', default='1-00:00:00')
@click.option('--nodelist', default=None)
@click.option('--cpus-per-task', default=None)
@click.argument('luigi_module')
@click.argument('luigi_task')
def slurm(partition, qos, mem_per_cpu, max_time, nodelist, cpus_per_task, luigi_module, luigi_task):
    env = Environment(loader=PackageLoader('qanta', 'slurm/templates'))
    template = env.get_template('luigi-template.sh.jinja2')
    sbatch_script = template.render({
        'luigi_module': luigi_module,
        'luigi_task': luigi_task,
        'partition': partition,
        'qos': qos,
        'mem_per_cpu': mem_per_cpu,
        'max_time': max_time,
        'nodelist': nodelist,
        'cpus_per_task': cpus_per_task
    })
    tmp_file = get_tmp_filename()
    with open(tmp_file, 'w') as f:
        f.write(sbatch_script)
    shell(f'sbatch {tmp_file}')
    shell(f'rm -f {tmp_file}')


@main.command()
@click.option('--generate-config/--no-generate-config', default=True, is_flag=True)
@click.option('--config-dir', default='.')
@click.option('--pid-file', default='elasticsearch.pid')
@click.argument('command', type=click.Choice(['start', 'stop', 'configure']))
def elasticsearch(generate_config, config_dir, pid_file, command):
    if generate_config:
        create_es_config(path.join(config_dir, 'elasticsearch.yml'))

    if command == 'configure':
        return

    if command == 'start':
        start_elasticsearch(config_dir, pid_file)
    elif command == 'stop':
        stop_elasticsearch(pid_file)


@main.command()
def answer_map_google_csvs():
    from qanta.ingestion.gspreadsheets import create_answer_mapping_csvs
    create_answer_mapping_csvs()


@main.command()
@click.argument('category_csv')
@click.argument('out_json')
def categorylinks_to_disambiguation(category_csv, out_json):
    disambiguation_pages = set()
    blacklist = {
        'Articles_with_links_needing_disambiguation_from_April_2018',
        'All_articles_with_links_needing_disambiguation'
    }
    with open(category_csv) as f:
        reader = csv.reader(f)
        for r in tqdm.tqdm(reader, mininterval=1):
            page_id, category = r[0], r[1]
            if ((category not in blacklist) and
                    ('disambiguation' in category.lower()) and
                    ('articles_with_links_needing_disambiguation' not in category)):
                disambiguation_pages.add(int(page_id))

    with open(out_json, 'w') as f:
        json.dump(list(disambiguation_pages), f)


@main.command()
@click.argument('csv_input')
@click.argument('json_dir')
def nonnaqt_to_json(csv_input, json_dir):
    question_sentences = defaultdict(list)
    with open(csv_input) as f:
        csv_rows = list(csv.reader(f))
        for r in csv_rows[1:]:
            if len(r) != 5:
                raise ValueError('Invalid csv row, must have 5 columns')
            qnum, sent, text, page, fold = r
            qnum = int(qnum)
            sent = int(sent)
            question_sentences[qnum].append({
                'qnum': qnum, 'sent': sent,
                'text': text, 'page': page, 'fold': fold
            })

    questions = []
    for sentences in tqdm.tqdm(question_sentences.values()):
        ordered_sentences = sorted(sentences, key=lambda s: s['sent'])
        text = ' '.join(s['text'] for s in ordered_sentences)
        tokenizations = []
        position = 0
        for i in range(len(ordered_sentences)):
            sent = ordered_sentences[i]['text']
            length = len(sent)
            tokenizations.append((position, position + length))
            position += length + 1
        q = ordered_sentences[0]
        questions.append({
            'answer': '',
            'category': '',
            'subcategory': '',
            'tournament': '',
            'year': -1,
            'dataset': 'non_naqt',
            'difficulty': '',
            'first_sentence': ordered_sentences[0]['text'],
            'qanta_id': q['qnum'],
            'fold': q['fold'],
            'gameplay': False,
            'page': q['page'],
            'proto_id': None,
            'qdb_id': None,
            'text': text,
            'tokenizations': tokenizations
        })

    train_questions = [q for q in questions if q['fold'] == 'guesstrain']
    dev_questions = [q for q in questions if q['fold'] == 'guessdev']
    test_questions = [q for q in questions if q['fold'] == 'test']
    for q in test_questions:
        q['fold'] = 'guesstest'

    from qanta.ingestion.preprocess import format_qanta_json
    from qanta.util.constants import DS_VERSION

    with open(path.join(json_dir, f'qanta.mapped.{DS_VERSION}.json'), 'w') as f:
        json.dump(format_qanta_json(questions, DS_VERSION), f)

    with open(path.join(json_dir, f'qanta.train.{DS_VERSION}.json'), 'w') as f:
        json.dump(format_qanta_json(train_questions, DS_VERSION), f)

    with open(path.join(json_dir, f'qanta.dev.{DS_VERSION}.json'), 'w') as f:
        json.dump(format_qanta_json(dev_questions, DS_VERSION), f)

    with open(path.join(json_dir, f'qanta.test.{DS_VERSION}.json'), 'w') as f:
        json.dump(format_qanta_json(test_questions, DS_VERSION), f)

    from sklearn.model_selection import train_test_split
    guess_train, guess_val = train_test_split(train_questions, random_state=42, train_size=.9)
    with open(path.join(json_dir, f'qanta.torchtext.train.{DS_VERSION}.json'), 'w') as f:
        json.dump(format_qanta_json(guess_train, DS_VERSION), f)

    with open(path.join(json_dir, f'qanta.torchtext.val.{DS_VERSION}.json'), 'w') as f:
        json.dump(format_qanta_json(guess_val, DS_VERSION), f)

    with open(path.join(json_dir, f'qanta.torchtext.dev.{DS_VERSION}.json'), 'w') as f:
        json.dump(format_qanta_json(dev_questions, DS_VERSION), f)


@main.command()
@click.argument('adversarial_input')
@click.argument('json_dir')
def adversarial_to_json(adversarial_input, json_dir):
    with open(adversarial_input) as f:
        have_question = False
        rows = []
        question = None
        for i, line in enumerate(f):
            if line == '\n':
                continue
            elif have_question:
                answer = line.strip().replace(' ', '_')
                rows.append({
                    'text': question,
                    'page': answer,
                    'answer': '',
                    'qanta_id': 1000000 + i,
                    'proto_id': None,
                    'qdb_id': None,
                    'category': '',
                    'subcategory': '',
                    'tournament': '',
                    'difficulty': '',
                    'dataset': 'adversarial',
                    'year': -1,
                    'fold': 'expo',
                    'gameplay': False
                })
                have_question = False
            else:
                question = line.strip()
                have_question = True

    from qanta.ingestion.preprocess import add_sentences_, format_qanta_json
    from qanta.util.constants import DS_VERSION
    add_sentences_(rows, parallel=False)
    with open(path.join(json_dir, f'qanta.expo.{DS_VERSION}.json'), 'w') as f:
        json.dump(format_qanta_json(rows, DS_VERSION), f)


if __name__ == '__main__':
    main()
