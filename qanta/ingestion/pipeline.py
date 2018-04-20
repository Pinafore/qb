import json
from os import path
from luigi import LocalTarget, Task, WrapperTask, Parameter

from qanta.util.io import shell, get_tmp_filename, safe_path
from qanta.pipeline.preprocess import WikipediaTitles, WikipediaRawRedirects
from qanta.ingestion.normalization import Protobowl, QuizdbOrg, merge_datasets, assign_folds
from qanta.ingestion.answer_mapping import create_answer_map, write_answer_map, unmapped_to_mapped_questions, questions_to_sqlite


DS_VERSION = '2018.04.18'
S3_HTTP_PREFIX = 'https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/'
DATASET_PREFIX = 'data/external/datasets'
QANTA_UNMAPPED_DATASET_PATH = f'data/external/datasets/qanta.unmapped.{DS_VERSION}.json'
QANTA_MAPPED_DATASET_PATH = f'data/external/datasets/qanta.mapped.{DS_VERSION}.json'
QANTA_MAPPED_SQL_DS_PATH = f'data/external/datasets/qanta.mapped.{DS_VERSION}.sqlite3'
ANSWER_MAP_PATH = 'data/external/answer_mapping/answer_map.json'
UNBOUND_ANSWER_PATH = 'data/external/answer_mapping/unbound_answers.json'


QDB_DATE = '04182018'
QDB_CATEGORIES = f'quizdb.org-{QDB_DATE}.categories.json'
QDB_SUBCATEGORIES = f'quizdb.org-{QDB_DATE}.subcategories.json'
QDB_TOURNAMENTS = f'quizdb.org-{QDB_DATE}.tournaments.json'
QDB_TOSSUPS = f'quizdb.org-{QDB_DATE}.tossups.json'

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


class CreateUnmappedQantaDataset(Task):
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
        assign_folds(qanta_questions)
        with open(safe_path(QANTA_UNMAPPED_DATASET_PATH), 'w') as f:
            json.dump(qanta_questions, f)

    def output(self):
        return LocalTarget(QANTA_UNMAPPED_DATASET_PATH)


class CreateAnswerMap(Task):
    def requires(self):
        yield CreateUnmappedQantaDataset()
        yield WikipediaRawRedirects()
        yield WikipediaTitles()

    def run(self):
        with open(QANTA_UNMAPPED_DATASET_PATH) as f:
            unmapped_qanta_questions = json.load(f)

        answer_map, unbound_answers = create_answer_map(unmapped_qanta_questions)
        write_answer_map(answer_map, unbound_answers, ANSWER_MAP_PATH, UNBOUND_ANSWER_PATH)

    def output(self):
        return [
            LocalTarget(ANSWER_MAP_PATH),
            LocalTarget(UNBOUND_ANSWER_PATH)
        ]


class CreateMappedQantaDataset(Task):
    def requires(self):
        yield CreateUnmappedQantaDataset()
        yield CreateAnswerMap()

    def run(self):
        with open(ANSWER_MAP_PATH) as f:
            answer_map = json.load(f)['answer_map']
        with open(QANTA_UNMAPPED_DATASET_PATH) as f:
            qanta_questions = json.load(f)

        unmapped_to_mapped_questions(qanta_questions, answer_map)
        with open(QANTA_MAPPED_DATASET_PATH, 'w') as f:
            json.dump(qanta_questions, f)

        tmp_db = get_tmp_filename()
        questions_to_sqlite(qanta_questions, tmp_db)
        shell(f'mv {tmp_db} {QANTA_MAPPED_SQL_DS_PATH}')

    def output(self):
        return [
            LocalTarget(QANTA_MAPPED_DATASET_PATH),
            LocalTarget(QANTA_MAPPED_SQL_DS_PATH)
        ]
