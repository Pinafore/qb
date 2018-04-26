#!/usr/bin/env python
import click
import subprocess
from os import path, makedirs


DS_VERSION = '2018.04.18'
S3_HTTP_PREFIX = 'https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/'
S3_AWS_PREFIX = 's3://pinafore-us-west-2/qanta-jmlr/datasets/'
LOCAL_PREFIX = 'data/external/datasets/'

QANTA_UNMAPPED_DATASET_PATH = f'qanta.unmapped.{DS_VERSION}.json'
QANTA_PROCESSED_DATASET_PATH = f'qanta.processed.{DS_VERSION}.json'
QANTA_MAPPED_DATASET_PATH = f'qanta.mapped.{DS_VERSION}.json'
QANTA_SQLITE_DATASET_PATH = f'qanta.{DS_VERSION}.sqlite3'
QANTA_TRAIN_DATASET_PATH = f'qanta.train.{DS_VERSION}.json'
QANTA_DEV_DATASET_PATH = f'qanta.dev.{DS_VERSION}.json'
QANTA_TEST_DATASET_PATH = f'qanta.test.{DS_VERSION}.json'

FILES = [
    QANTA_UNMAPPED_DATASET_PATH,
    QANTA_PROCESSED_DATASET_PATH,
    QANTA_MAPPED_DATASET_PATH,
    QANTA_SQLITE_DATASET_PATH,
    QANTA_TRAIN_DATASET_PATH,
    QANTA_DEV_DATASET_PATH,
    QANTA_TEST_DATASET_PATH
]


def make_file_pairs(source_prefix, target_prefix):
    return [(path.join(source_prefix, f), path.join(target_prefix, f)) for f in FILES]


def shell(command):
    return subprocess.run(command, check=True, shell=True, stderr=subprocess.STDOUT)


@click.group()
def main():
    pass


@main.command()
def download():
    makedirs(LOCAL_PREFIX, exist_ok=True)
    for s3_file, local_file in make_file_pairs(S3_HTTP_PREFIX, LOCAL_PREFIX):
        print(f'Downloading {s3_file} to {local_file}')
        shell(f'wget -O {local_file} {s3_file}')


@main.command()
def upload():
    for local_file, s3_file in make_file_pairs(LOCAL_PREFIX, S3_AWS_PREFIX):
        print(f'Uploading {local_file} to {s3_file}')
        shell(f'aws s3 cp {local_file} {s3_file}')


if __name__ == '__main__':
    main()