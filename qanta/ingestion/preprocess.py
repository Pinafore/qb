import sqlite3
import spacy
from qanta.spark import create_spark_context

# Each spark worker needs to load its own copy of the NLP model
# separately since its not serializable and thus not broadcastable
nlp_ref = []


def nlp(text):
    if len(nlp_ref) == 0:
        nlp_ref.append(spacy.load('en_core_web_lg'))

    if len(nlp_ref) == 1:
        doc = nlp_ref[0](text)
        sents = list(doc.sents)
        return sents[0].end_char
    else:
        raise ValueError('There should be exactly one nlp model per spark worker')


def format_qanta_json(questions, version):
    return {
        'questions': questions,
        'version': version,
        'maintainer_name': 'Pedro Rodriguez',
        'maintainer_contact': 'pedro@snowgeek.org',
        'maintainer_website': 'http://pedrorodriguez.io',
        'project_website': 'https://github.com/pinafore/qb'
    }


def add_first_sentence(questions):
    text_questions = [q['text'] for q in questions]
    sc = create_spark_context()
    first_sent_end_chars = sc.parallelize(text_questions, 4000).map(nlp).collect()
    first_sentences = [q[:end_char] for q, end_char in zip(text_questions, first_sent_end_chars)]
    for q, sent, pos in zip(questions, first_sentences, first_sent_end_chars):
        q['first_sentence'] = sent
        q['first_end_char'] = pos


def questions_to_sqlite(qanta_questions, db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('DROP TABLE IF EXISTS questions;')
    c.execute("""
        CREATE TABLE questions (
          qanta_id INT PRIMARY KEY NOT NULL,
          "text" TEXT NOT NULL, first_sentence TEXT NOT NULL, first_end_char INT NOT NULL,
          answer TEXT NOT NULL, page TEXT,
          fold TEXT NOT NULL,
          category TEXT, subcategory TEXT,
          tournament TEXT, difficulty TEXT, year INT,
          proto_id INT, qdb_id INT, dataset TEXT NOT NULL
        )
    """)
    c.executemany(
        'INSERT INTO questions values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
        [(
            q['qanta_id'], q['text'], q['first_sentence'], q['first_end_char'],
            q['answer'], q['page'], q['fold'],
            q['category'], q['subcategory'], q['tournament'], q['difficulty'],
            q['year'], q['proto_id'], q['qdb_id'], q['dataset']
        ) for q in qanta_questions]
    )
    conn.commit()
    conn.close()