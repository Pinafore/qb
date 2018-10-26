import json
from os import path
from luigi import LocalTarget, Task, WrapperTask, Parameter
import yaml

from sklearn.model_selection import train_test_split
from qanta.util.io import shell, get_tmp_filename, safe_path, safe_open
from qanta.util.constants import (
    DATASET_PREFIX, DS_VERSION, QANTA_MAP_REPORT_PATH,
    QANTA_MAPPED_DATASET_PATH, QANTA_SQL_DATASET_PATH,
    QANTA_TRAIN_DATASET_PATH, QANTA_DEV_DATASET_PATH, QANTA_TEST_DATASET_PATH,
    QANTA_TORCH_TRAIN_LOCAL_PATH, QANTA_TORCH_VAL_LOCAL_PATH, QANTA_TORCH_DEV_LOCAL_PATH,
    GUESSER_TRAIN_FOLD, GUESSER_DEV_FOLD
)
from qanta.pipeline.preprocess import WikipediaTitles, WikipediaRawRedirects
from qanta.ingestion.normalization import Protobowl, QuizdbOrg, merge_datasets, assign_folds_
from qanta.ingestion.answer_mapping import create_answer_map, write_answer_map, unmapped_to_mapped_questions
from qanta.ingestion.annotated_mapping import PageAssigner
from qanta.ingestion.preprocess import format_qanta_json, add_sentences_, questions_to_sqlite
from qanta.ingestion.protobowl import compute_question_player_counts


S3_HTTP_PREFIX = 'https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/'
QANTA_UNMAPPED_DATASET_PATH = path.join(DATASET_PREFIX, f'qanta.unmapped.{DS_VERSION}.json')
QANTA_PREPROCESSED_DATASET_PATH = path.join(DATASET_PREFIX, f'qanta.processed.{DS_VERSION}.json')
QANTA_FOLDED_DATASET_PATH = path.join(DATASET_PREFIX, f'qanta.folded.{DS_VERSION}.json')

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

PROTOBOWL_LOGS = 'protobowl-042818.log'
PROTOBOWL_LOGS_PATH = path.join(DATASET_PREFIX, 'protobowl', PROTOBOWL_LOGS)

PROTOBOWL_QUESTION_PLAYER_COUNTS = path.join(DATASET_PREFIX, 'protobowl', 'question_player_counts.json')


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
        yield Download(
            url=path.join(S3_HTTP_PREFIX, PROTOBOWL_LOGS),
            path=safe_path(PROTOBOWL_LOGS_PATH)
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
        with open(safe_path(QANTA_UNMAPPED_DATASET_PATH), 'w') as f:
            json.dump(format_qanta_json(qanta_questions, DS_VERSION), f)

    def output(self):
        return LocalTarget(QANTA_UNMAPPED_DATASET_PATH)


class CreateProcessedQantaDataset(Task):
    def requires(self):
        yield CreateUnmappedQantaDataset()

    def run(self):
        with open(QANTA_UNMAPPED_DATASET_PATH) as f:
            qanta_questions = json.load(f)['questions']
        add_sentences_(qanta_questions)
        with open(QANTA_PREPROCESSED_DATASET_PATH, 'w') as f:
            json.dump(format_qanta_json(qanta_questions, DS_VERSION), f)

    def output(self):
        return LocalTarget(QANTA_PREPROCESSED_DATASET_PATH)


class CreateAnswerMap(Task):
    def requires(self):
        yield CreateProcessedQantaDataset()
        yield WikipediaRawRedirects()
        yield WikipediaTitles()

    def run(self):
        with open(QANTA_PREPROCESSED_DATASET_PATH) as f:
            unmapped_qanta_questions = json.load(f)['questions']

        answer_map, amb_answer_map, unbound_answers, report = create_answer_map(unmapped_qanta_questions)
        with safe_open('data/external/answer_mapping/automatic_report.json', 'w') as f:
            json.dump(report, f)
        write_answer_map(answer_map, amb_answer_map, unbound_answers, ANSWER_MAP_PATH, UNBOUND_ANSWER_PATH)

    def output(self):
        return (
            LocalTarget(ANSWER_MAP_PATH),
            LocalTarget(UNBOUND_ANSWER_PATH),
            LocalTarget('data/external/answer_mapping/automatic_report.json')
        )


class CreateProtobowlQuestionPlayerCounts(Task):
    def requires(self):
        yield DownloadProtobowl()

    def run(self):
        question_player_counts = compute_question_player_counts(PROTOBOWL_LOGS_PATH)
        with open(PROTOBOWL_QUESTION_PLAYER_COUNTS, 'w') as f:
            json.dump(question_player_counts, f)

    def output(self):
        return LocalTarget(PROTOBOWL_QUESTION_PLAYER_COUNTS)


class CreateFoldedQantaDataset(Task):
    def requires(self):
        yield CreateProcessedQantaDataset()
        yield CreateProtobowlQuestionPlayerCounts()

    def run(self):
        with open(QANTA_PREPROCESSED_DATASET_PATH) as f:
            qanta_questions = json.load(f)['questions']

        with open(PROTOBOWL_QUESTION_PLAYER_COUNTS) as f:
            question_player_counts = json.load(f)
        assign_folds_(qanta_questions, question_player_counts)

        with open(QANTA_FOLDED_DATASET_PATH, 'w') as f:
            json.dump(format_qanta_json(qanta_questions, DS_VERSION), f)

    def output(self):
        return LocalTarget(QANTA_FOLDED_DATASET_PATH)


class CreateMappedQantaDataset(Task):
    def requires(self):
        yield CreateFoldedQantaDataset()
        yield CreateAnswerMap()
        yield WikipediaTitles()

    def run(self):
        with open(ANSWER_MAP_PATH) as f:
            content = json.load(f)
            answer_map = content['answer_map']
            ambig_answer_map = content['ambig_answer_map']
        with open(QANTA_FOLDED_DATASET_PATH) as f:
            qanta_questions = json.load(f)['questions']

        with open('data/internal/page_assignment/unmappable.yaml') as f:
            unmappable = yaml.load(f)

        page_assigner = PageAssigner()
        mapping_report = unmapped_to_mapped_questions(
            qanta_questions,
            answer_map, ambig_answer_map,
            unmappable, page_assigner
        )

        with open(QANTA_MAPPED_DATASET_PATH, 'w') as f:
            json.dump(format_qanta_json(qanta_questions, DS_VERSION), f)

        with open(QANTA_MAP_REPORT_PATH, 'w') as f:
            json.dump(mapping_report, f)

    def output(self):
        return LocalTarget(QANTA_MAPPED_DATASET_PATH), LocalTarget(QANTA_MAP_REPORT_PATH)


class GenerateSqliteDB(Task):
    def requires(self):
        yield CreateMappedQantaDataset()

    def run(self):
        with open(QANTA_MAPPED_DATASET_PATH) as f:
            qanta_questions = json.load(f)['questions']

        tmp_db = get_tmp_filename()
        questions_to_sqlite(qanta_questions, tmp_db)
        shell(f'mv {tmp_db} {QANTA_SQL_DATASET_PATH}')

    def output(self):
        return LocalTarget(QANTA_SQL_DATASET_PATH)


class FilterAndPartitionQantaDataset(Task):
    def requires(self):
        yield CreateMappedQantaDataset()
        yield CreateProtobowlQuestionPlayerCounts()

    def run(self):
        with open(QANTA_MAPPED_DATASET_PATH) as f:
            questions = [q for q in json.load(f)['questions'] if q['page'] is not None]
        train_questions = [q for q in questions if 'train' in q['fold']]
        dev_questions = [q for q in questions if 'dev' in q['fold']]
        test_questions = [q for q in questions if 'test' in q['fold']]

        with open(QANTA_TRAIN_DATASET_PATH, 'w') as f:
            json.dump(format_qanta_json(train_questions, DS_VERSION), f)

        with open(QANTA_DEV_DATASET_PATH, 'w') as f:
            json.dump(format_qanta_json(dev_questions, DS_VERSION), f)

        with open(QANTA_TEST_DATASET_PATH, 'w') as f:
            json.dump(format_qanta_json(test_questions, DS_VERSION), f)

    def output(self):
        return [
            LocalTarget(QANTA_TRAIN_DATASET_PATH),
            LocalTarget(QANTA_DEV_DATASET_PATH),
            LocalTarget(QANTA_TEST_DATASET_PATH)
        ]


class TorchTextDataset(Task):
    def requires(self):
        yield FilterAndPartitionQantaDataset()

    def run(self):
        with open(QANTA_TRAIN_DATASET_PATH) as f:
            all_guess_train = [q for q in json.load(f)['questions'] if q['fold'] == GUESSER_TRAIN_FOLD]

        guess_train, guess_val = train_test_split(all_guess_train, random_state=42, train_size=.9)

        with open(QANTA_DEV_DATASET_PATH) as f:
            guess_dev = [q for q in json.load(f)['questions'] if q['fold'] == GUESSER_DEV_FOLD]

        with open(QANTA_TORCH_TRAIN_LOCAL_PATH, 'w') as f:
            json.dump(format_qanta_json(guess_train, DS_VERSION), f)

        with open(QANTA_TORCH_VAL_LOCAL_PATH, 'w') as f:
            json.dump(format_qanta_json(guess_val, DS_VERSION), f)

        with open(QANTA_TORCH_DEV_LOCAL_PATH, 'w') as f:
            json.dump(format_qanta_json(guess_dev, DS_VERSION), f)

    def output(self):
        return [
            LocalTarget(QANTA_TORCH_TRAIN_LOCAL_PATH),
            LocalTarget(QANTA_TORCH_VAL_LOCAL_PATH),
            LocalTarget(QANTA_TORCH_DEV_LOCAL_PATH)
        ]


class QantaDataset(WrapperTask):
    def requires(self):
        yield FilterAndPartitionQantaDataset()
        yield GenerateSqliteDB()
        yield TorchTextDataset()


class TrickMeDataset(Task):
    def requires(self):
        yield QantaDataset()

    def run(self):
        pass
