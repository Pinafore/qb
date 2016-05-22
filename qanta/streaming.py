import subprocess
import socket
import time
import requests
import json
from typing import List, Tuple
from collections import namedtuple

from qanta.util.environment import QB_QUESTION_DB, QB_SPARK_MASTER, QB_STREAMING_CORES
from qanta.util.qdb import QuestionDatabase
from qanta.extract_features import instantiate_feature

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from functional import seq, pseq

from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext

DB_URL = 'postgresql+psycopg2://postgres:postgres@localhost:5432'
Base = declarative_base()
engine = create_engine(DB_URL)
SessionFactory = sessionmaker(bind=engine)
StreamGuess = namedtuple('StreamGuess', 'id text guess features score')

QBApiQuestion = namedtuple('QBApiQuestion', 'fold id word_count position text guess')


class Question(Base):
    __tablename__ = 'questions'

    id = Column(Integer, primary_key=True)
    text = Column(String)
    response = Column(String, nullable=True)
    external_id = Column(Integer)

    def __repr__(self):
        fmtstr = '<Question(id={id}, text={text}, response={response} external_id={external_id}'
        return fmtstr.format(
            id=self.id, text=self.text, response=self.response, external_id=self.external_id)

    def get_buzz(self):
        guesses = json.loads(self.response)
        top_guess = guesses[0]
        print("top guess: {0}".format(top_guess))
        return top_guess[0] > 0, top_guess[1]


class QuestionManager:
    def __init__(self, questions: List[Tuple[int, str]]):
        self.questions = questions

    def get(self) -> List[Tuple[int, str]]:
        questions = self.questions
        self.questions = []
        return questions

    def update(self, question_responses: List[Question]) -> None:
        pass

    def submit(self):
        pass


class QBApiQuestionManager(QuestionManager):
    def __init__(self, domain: str, user_id: int, api_key: str):
        super().__init__([])
        self.domain = domain
        self.user_id = user_id
        self.api_key = api_key
        self.questions = self.get_api_questions()
        self.buzzed_questions = seq([])

    def get_api_questions(self):
        url = 'http://{domain}/qb-api/v1/questions'.format(domain=self.domain)
        response = seq(requests.get(url).json()['questions'])
        return pseq(response)\
            .map(lambda r: QBApiQuestion(**r, position=-1, text='', guess=None)).cache()

    def request_text(self, q_id, position) -> str:
        url = 'http://{domain}/qb-api/v1/question/{q_id}/{position}'.format(
            domain=self.domain, q_id=q_id, position=position)
        response = requests.post(url, data={'user_id': self.user_id, 'api_key': self.api_key})
        data = response.json()['word']
        return data['position'], data['text']

    def update_questions(self):
        def update_question_tuple(q: QBApiQuestion):
            position, text = self.request_text(q.id, q.position + 1)
            if q.text != '':
                text = q.text + text
            return QBApiQuestion(q.fold, q.id, q.word_count, position, text, q.guess)
        self.questions = pseq(self.questions).map(update_question_tuple).cache()

    def get(self) -> List[Tuple[int, str]]:
        return self.questions.map(lambda q: (q.id, q.text)).list()

    def update(self, question_responses: List[Question]):
        print("Updating questions")
        self.update_questions()

        print("Length of questions after update: {0}".format(self.questions.len()))

        def merge(record: Tuple[int, Tuple[QBApiQuestion, Question]]) -> QBApiQuestion:
            api_question = record[1][0]
            response = record[1][1]
            if response is None:
                return api_question
            else:
                buzz, guess = response.get_buzz()
                print("Recieved buzz: {0}".format(buzz))
                if buzz or api_question.position + 1 == api_question.word_count:
                    return QBApiQuestion(
                        api_question.fold,
                        api_question.id,
                        api_question.word_count,
                        api_question.position,
                        api_question.text,
                        guess)
                else:
                    return api_question
        keyed_questions = self.questions.map(lambda q: (q.id, q))
        keyed_responses = seq(question_responses).map(lambda r: (r.external_id, r))
        merged_questions = keyed_questions.left_join(keyed_responses).map(merge).cache()
        print("Length of merged questions: {0}".format(merged_questions.len()))
        self.questions, self.buzzed_questions = merged_questions.partition(
            lambda q: q.guess is None)
        print("Length of questions: {0}".format(self.questions.len()))
        print("Length of buzzed questions: {0}".format(self.buzzed_questions.len()))

    def submit(self):
        print("Buzzed questions")
        print(self.buzzed_questions)
        print("Non-buzzed questions (should be empty)")
        print(self.questions)


def create_sc():
    spark_conf = SparkConf()
    spark_conf = spark_conf \
        .set('spark.max.cores', QB_STREAMING_CORES) \
        .set('spark.executor.cores', QB_STREAMING_CORES)
    return SparkContext(appName='Quiz Bowl Streaming', master=QB_SPARK_MASTER, conf=spark_conf)


def vw_score(stream_guesses):
    print("Scoring with VW")
    output = []
    vw_input = '\n'.join(sg.features for sg in stream_guesses).encode('utf-8')
    out = subprocess.run(['nc', 'localhost', '26542'], input=vw_input, stdout=subprocess.PIPE)
    if len(out.stdout) != 0:
        out_lines = out.stdout.decode('utf-8').split('\n')
        for sg, line in zip(stream_guesses, out_lines):
            if len(line) != 0:
                score = float(line.split()[0])
                output.append(StreamGuess(sg.id, sg.text, sg.guess, sg.features, score))
    print("Done with VW Scoring")
    return output


def generate_guesses(line: str, b_features):
    features = b_features.value
    streaming_id, text = line.split('|')
    streaming_id = int(streaming_id)
    deep_feature = features['deep']
    guesses = deep_feature.text_guess([text])

    return [StreamGuess(streaming_id, text, g, None, None) for g in guesses.keys()]


def evaluate_features(stream_guess: StreamGuess, b_features):
    features = b_features.value
    row = ''
    for name in ['label', 'deep']:
        feature_text = features[name].vw_from_title(stream_guess.guess, stream_guess.text)
        if name == 'label':
            row = feature_text
        else:
            row += ' ' + feature_text
    return StreamGuess(stream_guess.id, stream_guess.text, stream_guess.guess, row, None)


def save(stream_guesses):
    session = SessionFactory()
    sg_by_id = seq(stream_guesses).group_by(lambda sg: sg.id)
    for sg_id, sg_list in sg_by_id:
        response = sorted([(sg.score, sg.guess) for sg in sg_list], reverse=True)
        q = session.query(Question).filter(Question.id == sg_id).first()
        q.response = json.dumps(response)
        session.commit()
    session.close()


def score_and_save(rdd):
    stream_guesses = rdd.collect()
    final_stream_guesses = vw_score(stream_guesses)
    save(final_stream_guesses)


def start_spark_streaming():
    question_db = QuestionDatabase(QB_QUESTION_DB)
    features = {name: instantiate_feature(name, question_db) for name in ['deep', 'label']}

    sc = create_sc()
    b_features = sc.broadcast(features)
    ssc = StreamingContext(sc, 1)

    ssc.socketTextStream('localhost', 9999) \
        .repartition(QB_STREAMING_CORES) \
        .flatMap(lambda line: generate_guesses(line, b_features)) \
        .map(lambda sg: evaluate_features(sg, b_features)) \
        .foreachRDD(score_and_save)

    ssc.start()
    ssc.awaitTermination()
    sc.stop()


def create_socket_connection():
    spark_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    spark_socket.bind(('', 9999))
    spark_socket.listen(1)
    connection, _ = spark_socket.accept()
    return connection


def get_buzzes(questions_text, spark_connection, session):
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    for external_id, text in questions_text:
        q = Question(text=text, response=None, external_id=external_id)
        session.add(q)
    session.commit()

    print("Sending questions to Spark")
    for q in session.query(Question):
        request = bytes('{id}|{text}\n'.format(id=q.id, text=q.text), 'utf8')
        spark_connection.sendall(request)

    print("Waiting for guesses from spark")
    while True:
        session.expire_all()
        responses = seq(session.query(Question).all())
        if responses.count(lambda q: q.response is not None) == len(questions_text):
            return responses.list()
        time.sleep(.1)


def start_qanta_streaming():
    print("Starting Qanta server")

    #qdb = QuestionDatabase(QB_QUESTION_DB)
    print("Waiting for spark to connect...")
    spark_connection = create_socket_connection()
    print("Connection established")

    session = SessionFactory()
    #questions = seq(qdb.all_questions().values()).filter(lambda q: q.fold == 'test')
    #questions_text = questions.map(lambda q: q.flatten_text()).take(10).enumerate().list()
    #manager = QuestionManager(questions_text)
    manager = QBApiQuestionManager(
        'ec2-54-153-32-151.us-west-1.compute.amazonaws.com',
        1,
        'DtNUllozIVvQkrtNAqWMaYdduKuzzZwFnVotdEAxgFywYZsUsvaZYXyuoXPtfNdV')
    manager.update([])

    current_questions = manager.get()
    i = 0
    while len(current_questions) != 0:
        print("Iteration {0} starting".format(i))
        buzzes = get_buzzes(current_questions, spark_connection, session)
        manager.update(buzzes)
        current_questions = manager.get()
        i += 1
    manager.submit()

    session.close()
    spark_connection.close()
