import subprocess
import socket
import time
from collections import namedtuple

from qanta.util.environment import QB_QUESTION_DB, QB_SPARK_MASTER, QB_STREAMING_CORES
from qanta.util.qdb import QuestionDatabase
from qanta.extract_features import instantiate_feature

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from functional import seq

from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext


DB_URL = 'postgresql+psycopg2://postgres:postgres@localhost:5432'
Base = declarative_base()
engine = create_engine(DB_URL)
SessionFactory = sessionmaker(bind=engine)
StreamGuess = namedtuple('StreamGuess', 'id text guess features score')


class Question(Base):
    __tablename__ = 'questions'

    id = Column(Integer, primary_key=True)
    text = Column(String)
    response = Column(String, nullable=True)

    def __repr__(self):
        return '<Question(id={id}, text={text}, response={response}'.format(
            id=self.id, text=self.text, response=self.response)


def create_sc():
    spark_conf = SparkConf()
    spark_conf = spark_conf\
        .set('spark.max.cores', QB_STREAMING_CORES)\
        .set('spark.executor.cores', QB_STREAMING_CORES)
    return SparkContext(appName='Quiz Bowl Streaming', master=QB_SPARK_MASTER, conf=spark_conf)


def vw_score(stream_guesses):
    output = []
    for sg in stream_guesses:
        out = subprocess.run(
            ['bash', '/home/ubuntu/qb/bin/vw-line.sh', sg.features], stdout=subprocess.PIPE)
        if len(out.stdout) != 0:
            score = float(out.stdout.split()[0])
            output.append(StreamGuess(sg.id, sg.text, sg.guess, sg.features, score))
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
            row += '\t' + feature_text
    return StreamGuess(stream_guess.id, stream_guess.text, stream_guess.guess, row, None)


def save(stream_guesses):
    session = SessionFactory()
    sg_by_id = seq(stream_guesses).group_by(lambda sg: sg.id)
    for sg_id, sg_list in sg_by_id:
        response = sorted([(sg.score, sg.guess) for sg in sg_list], reverse=True)
        q = session.query(Question).filter(Question.id == sg_id).first()
        q.response = str(response)
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
        .repartition(QB_STREAMING_CORES)\
        .flatMap(lambda line: generate_guesses(line, b_features))\
        .map(lambda sg: evaluate_features(sg, b_features))\
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

    for text in questions_text:
        q = Question(text=text, response=None)
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

    qdb = QuestionDatabase(QB_QUESTION_DB)
    questions = seq(qdb.all_questions().values()).filter(lambda q: q.fold == 'test')
    session = SessionFactory()
    print("Waiting for spark to connect...")
    spark_connection = create_socket_connection()
    print("Connection established")

    questions_text = questions.map(lambda q: q.flatten_text()).take(10).list()
    buzzes = get_buzzes(questions_text, spark_connection, session)
    correct = 0
    print("Calculating accuracy")
    for q, b in zip(questions, buzzes):
        guess_list = eval(b)
        if guess_list[0][0] > 0 and guess_list[0][1] == q.page:
            correct += 1
    print("Accuracy: {0}".format(correct / len(questions_text)))

    session.close()
    spark_connection.close()
