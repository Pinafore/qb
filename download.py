#!/usr/bin/env python
import click
import subprocess
from os import path, makedirs


DS_VERSION = '2018.04.18'
S3_HTTP_PREFIX = 'https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/'
LOCAL_PREFIX = 'data/external/datasets/'

QANTA_UNMAPPED_DATASET_PATH = f'qanta.unmapped.{DS_VERSION}.json'
QANTA_PROCESSED_DATASET_PATH = f'qanta.processed.{DS_VERSION}.json'
QANTA_MAPPED_DATASET_PATH = f'qanta.mapped.{DS_VERSION}.json'
QANTA_SQLITE_DATASET_PATH = f'qanta.{DS_VERSION}.sqlite3'
QANTA_TRAIN_DATASET_PATH = f'qanta.train.{DS_VERSION}.json'
QANTA_DEV_DATASET_PATH = f'qanta.dev.{DS_VERSION}.json'
QANTA_TEST_DATASET_PATH = f'qanta.test.{DS_VERSION}.json'

FILES = [
    (path.join(S3_HTTP_PREFIX, QANTA_UNMAPPED_DATASET_PATH), path.join(LOCAL_PREFIX, QANTA_UNMAPPED_DATASET_PATH)),
    (path.join(S3_HTTP_PREFIX, QANTA_PROCESSED_DATASET_PATH), path.join(LOCAL_PREFIX, QANTA_PROCESSED_DATASET_PATH)),
    (path.join(S3_HTTP_PREFIX, QANTA_MAPPED_DATASET_PATH), path.join(LOCAL_PREFIX, QANTA_MAPPED_DATASET_PATH)),
    (path.join(S3_HTTP_PREFIX, QANTA_SQLITE_DATASET_PATH), path.join(LOCAL_PREFIX, QANTA_SQLITE_DATASET_PATH)),
    (path.join(S3_HTTP_PREFIX, QANTA_TRAIN_DATASET_PATH), path.join(LOCAL_PREFIX, QANTA_TRAIN_DATASET_PATH)),
    (path.join(S3_HTTP_PREFIX, QANTA_DEV_DATASET_PATH), path.join(LOCAL_PREFIX, QANTA_DEV_DATASET_PATH)),
    (path.join(S3_HTTP_PREFIX, QANTA_TEST_DATASET_PATH), path.join(LOCAL_PREFIX, QANTA_TEST_DATASET_PATH))
]


def shell(command):
    return subprocess.run(command, check=True, shell=True, stderr=subprocess.STDOUT)


@click.command()
def main():
    makedirs(LOCAL_PREFIX, exist_ok=True)
    for s3_file, local_file in FILES:
        shell(f'wget -O {local_file} {s3_file}')


if __name__ == '__main__':
    main()