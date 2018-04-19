import json
from os import path
from luigi import LocalTarget, Task, WrapperTask, Parameter

from qanta.util.io import shell, get_tmp_filename, safe_path
from qanta.ingestion.normalization import Protobowl, QuizdbOrg, merge_datasets


S3_HTTP_PREFIX = 'https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/'
DATASET_PREFIX = 'data/external/datasets'
QANTA_DATASET_PATH = 'data/external/datasets/qanta.2018.04.18.json'


QDB_CATEGORIES = 'quizdb.org-04182018.categories.json'
QDB_SUBCATEGORIES = 'quizdb.org-04182018.subcategories.json'
QDB_TOURNAMENTS = 'quizdb.org-04182018.tournaments.json'
QDB_TOSSUPS = 'quizdb.org-04182018.tossups.json'

QDB_CATEGORIES_PATH = path.join(DATASET_PREFIX, 'quizdb', QDB_CATEGORIES)
QDB_SUBCATEGORIES_PATH = path.join(DATASET_PREFIX, 'quizdb', QDB_SUBCATEGORIES)
QDB_TOURNAMENTS_PATH = path.join(DATASET_PREFIX, 'quizdb', QDB_TOURNAMENTS)
QDB_TOSSUPS_PATH = path.join(DATASET_PREFIX, 'quizdb', QDB_TOSSUPS)

PROTOBOWL_TOSSUPS = 'protobowl-05052017.json'
PROTOBOWL_TOSSUPS_PATH = path.join(DATASET_PREFIX, 'protobowl', PROTOBOWL_TOSSUPS)


class Download(Task):
    url = Parameter()  # type: str
    path = Parameter()  # type: str

    def run(self):
        tmp_file = get_tmp_filename()
        shell(f'wget {self.url} -O {tmp_file}')
        shell(f'mv {tmp_file} {self.path}')
        shell(f'rm -f {tmp_file}')

    def output(self):
        return LocalTarget(self.path)


class DownloadProtobowl(WrapperTask):
    def requires(self):
        yield Download(
            url=path.join(S3_HTTP_PREFIX, PROTOBOWL_TOSSUPS),
            path=safe_path(PROTOBOWL_TOSSUPS_PATH)
        )


class DownloadQuizdbOrg(WrapperTask):
    def requires(self):
        yield Download(url=path.join(S3_HTTP_PREFIX, QDB_CATEGORIES), path=safe_path(QDB_CATEGORIES_PATH))
        yield Download(url=path.join(S3_HTTP_PREFIX, QDB_SUBCATEGORIES), path=safe_path(QDB_SUBCATEGORIES_PATH))
        yield Download(url=path.join(S3_HTTP_PREFIX, QDB_TOURNAMENTS), path=safe_path(QDB_TOURNAMENTS_PATH))
        yield Download(url=path.join(S3_HTTP_PREFIX, QDB_TOSSUPS), path=safe_path(QDB_TOSSUPS_PATH))


class DownloadDatasets(WrapperTask):
    def requires(self):
        yield DownloadProtobowl()
        yield DownloadQuizdbOrg()


class MergeDatasets(Task):
    def requires(self):
        yield DownloadDatasets()

    def run(self):
        protobowl_questions = Protobowl.parse_tossups(PROTOBOWL_TOSSUPS_PATH)
        quizdb_tournaments = QuizdbOrg.parse_tournaments(QDB_TOURNAMENTS_PATH)
        quizdb_categories = QuizdbOrg.parse_categories(QDB_CATEGORIES_PATH)
        quizdb_subcategories = QuizdbOrg.parse_subcategories(QDB_SUBCATEGORIES_PATH)
        quizdb_questions = QuizdbOrg.parse_tossups(
            quizdb_tournaments, quizdb_categories, quizdb_subcategories, QDB_TOSSUPS_PATH
        )
        qanta_questions = merge_datasets(protobowl_questions, quizdb_questions)
        with open(safe_path(QANTA_DATASET_PATH), 'w') as f:
            json.dump(qanta_questions, f)

    def output(self):
        return LocalTarget(QANTA_DATASET_PATH)
