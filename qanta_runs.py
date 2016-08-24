#!/usr/bin/env python3

import os
from datetime import datetime
import boto3
import click
import sh


class S3:
    def __init__(self, bucket):
        self.s3 = boto3.resource('s3')
        self.bucket = bucket

    def list_runs(self):
        response = self.s3.meta.client.list_objects_v2(
            Bucket=self.bucket,
            Delimiter='/'
        )
        for f in response['CommonPrefixes']:
            yield f['Prefix'].replace('/', '')

    def create_run(self, date):
        if not os.path.exists('/tmp/qb'):
            os.makedirs('/tmp/qb')

        with open('/tmp/qb/run_id', 'w') as f:
            f.write(date)

        self.s3.meta.client.upload_file('/tmp/qb/run_id', self.bucket, '{}/run_id'.format(date))


@click.group()
@click.option('--bucket', help='AWS S3 bucket to checkpoint and restore from')
@click.pass_context
def cli(ctx, bucket):
    if bucket is None and os.environ.get('QB_AWS_S3_BUCKET') is not None:
        bucket = os.environ.get('QB_AWS_S3_BUCKET')
    elif bucket is None and os.environ.get('QB_AWS_S3_BUCKET') is None:
        raise ValueError('You must set QB_AWS_S3_BUCKET or pass --bucket as an argument')
    ctx.obj['s3'] = S3(bucket)


@cli.command()
@click.pass_context
def list(ctx):
    for key in ctx.obj['s3'].list_runs():
        print(key)


@cli.command()
@click.pass_context
def latest(ctx):
    latest_id = max([datetime.strptime(date, '%Y-%m-%d') for date in ctx.obj['s3'].list_runs()])
    print(latest_id.strftime('%Y-%m-%d'))


@cli.command()
@click.option('--date', help='Date to use for run identifier in YYYY-MM-DD format')
@click.pass_context
def create(ctx, date):
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    ctx.obj['s3'].create_run(date)


if __name__ == '__main__':
    cli(obj={})
