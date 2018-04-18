#!/usr/bin/env python3

import os
from urllib import parse
import subprocess
from datetime import datetime
import boto3
import click


PRE_PROCESS_TARGETS = {
    'data/external/wikipedia/cache_SUCCESS',
    'data/external/wikipedia/dump_redirects.pickle',
    'data/external/wikipedia/pages',
    'data/external/wikipedia/parsed-wiki',
    'data/external/wikipedia/parsed-wiki_SUCCESS',
    'data/external/wikipedia/wikipedia-titles.pickle',
    'data/external/wikipedia/all_wiki_redirects.csv',
    'data/external/wikidata_instance-of.pickle'
}


GUESS_TARGETS = {
    'output/guesser'
}

VW_INPUT = {'output/vw_input'}

VW_MODELS = {'output/models'}

PREDICTIONS = {'output/predictions'}

SUMMARIES = {'output/summary'}

REPORTING = {'output/reporting'}

EXPO = {'output/expo'}


CHECKPOINT_TARGETS = (
    PRE_PROCESS_TARGETS | GUESS_TARGETS |
    VW_INPUT | VW_MODELS | PREDICTIONS | SUMMARIES | REPORTING | EXPO
)

TARGET_GROUPS = {
    'preprocess': PRE_PROCESS_TARGETS,
    'guesser': GUESS_TARGETS,
    'vw_input': VW_INPUT,
    'vw_models': VW_MODELS,
    'predictions': PREDICTIONS,
    'summaries': SUMMARIES,
    'reporting': REPORTING,
    'expo': EXPO,
    'all': CHECKPOINT_TARGETS
}


CHECKPOINT_CHOICES = set(TARGET_GROUPS.keys()) | CHECKPOINT_TARGETS


class S3:
    def __init__(self, bucket, namespace):
        self.s3 = boto3.resource('s3')
        self.bucket = bucket
        self.namespace = namespace

    def list_runs(self):
        response = self.s3.meta.client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=self.namespace + '/',
            Delimiter='/'
        )
        for f in response['CommonPrefixes']:
            yield f['Prefix'].split('/')[1]

    def create_run(self, date):
        if not os.path.exists('/tmp/qb'):
            os.makedirs('/tmp/qb')

        with open('/tmp/qb/run_id', 'w') as f:
            f.write(date)

        self.s3.meta.client.upload_file(
            '/tmp/qb/run_id',
            self.bucket,
            '{}/{}/run_id'.format(self.namespace, date)
        )

    def latest_run(self):
        all_runs = [datetime.strptime(date, '%Y-%m-%d') for date in self.list_runs()]
        if len(all_runs) == 0:
            raise ValueError('There are no runs so therefore there is no latest run')
        latest_id = max(all_runs)
        return latest_id.strftime('%Y-%m-%d')


def fetch(variable, environment_variable):
    if variable is None:
        env_variable = os.environ.get(environment_variable)
        if env_variable is not None and env_variable != "":
            return env_variable
        else:
            raise ValueError('You must set {} or pass the variable as an option'.format(
                environment_variable))
    else:
        return variable


def shell(command):
    return subprocess.run(command, check=True, shell=True)


def compile_targets(targets):
    compiled_targets = set()
    for t in targets:
        if t in TARGET_GROUPS:
            compiled_targets |= TARGET_GROUPS[t]
        else:
            compiled_targets |= {t}

    return compiled_targets


@click.group()
@click.option('--bucket', help='AWS S3 bucket to checkpoint and restore from')
@click.option('--namespace', help='Namespace within bucket to checkpoint and restore from')
@click.pass_context
def cli(ctx, bucket, namespace):
    if not os.path.exists('/tmp/qb'):
        os.makedirs('/tmp/qb')

    ctx.obj['s3'] = S3(
        fetch(bucket, 'QB_AWS_S3_BUCKET'),
        fetch(namespace, 'QB_AWS_S3_NAMESPACE')
    )


@cli.command(name='list')
@click.pass_context
def list_runs(ctx):
    for key in sorted(ctx.obj['s3'].list_runs()):
        print(key)


@cli.command()
@click.pass_context
def latest(ctx):
    print(ctx.obj['s3'].latest_run())


@cli.command()
@click.pass_context
def keys(ctx):
    for k in sorted(TARGET_GROUPS):
        print(k)


@cli.command()
@click.option('--date', help='Date to use for run identifier in YYYY-MM-DD format')
@click.pass_context
def create(ctx, date):
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    ctx.obj['s3'].create_run(date)


@cli.command()
@click.option('--date', help="Which date to save the qanta run to, by default the most recent")
@click.argument('targets', nargs=-1, type=click.Choice(CHECKPOINT_CHOICES), required=True)
@click.pass_context
def save(ctx, date, targets):
    s3 = ctx.obj['s3']
    if date is None:
        date = s3.latest_run()

    for t in compile_targets(targets):
        name = parse.quote_plus(t)
        shell('tar cvf - {target} | lz4 > /tmp/qb/{name}.tar.lz4'.format(target=t, name=name))
        shell('aws s3 cp /tmp/qb/{name}.tar.lz4 s3://{bucket}/{namespace}/{date}/{name}'.format(
            name=name,
            bucket=s3.bucket,
            namespace=s3.namespace,
            date=date
        ))
        shell('rm /tmp/qb/{name}.tar.lz4'.format(name=name))


@cli.command()
@click.option('--date', help="Which date to restore the qanta run from")
@click.argument('targets', nargs=-1, type=click.Choice(CHECKPOINT_CHOICES), required=True)
@click.pass_context
def restore(ctx, date, targets):
    s3 = ctx.obj['s3']
    if date is None:
        date = s3.latest_run()

    for t in compile_targets(targets):
        name = parse.quote_plus(t)
        shell('aws s3 cp s3://{bucket}/{namespace}/{date}/{name} /tmp/qb/{name}.tar.lz4'.format(
            name=name,
            bucket=s3.bucket,
            namespace=s3.namespace,
            date=date
        ))
        shell('lz4 -d /tmp/qb/{name}.tar.lz4 | tar -x -C .'.format(name=name))
        shell('rm /tmp/qb/{name}.tar.lz4'.format(name=name))


if __name__ == '__main__':
    cli(obj={})
