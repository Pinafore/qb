import sqlite3
import spacy
import unidecode
from qanta import qlogging
from qanta.spark import create_spark_context


log = qlogging.get(__name__)

# Each spark worker needs to load its own copy of the NLP model
# separately since its not serializable and thus not broadcastable
nlp_ref = []


def nlp(text):
    if len(nlp_ref) == 0:
        nlp_ref.append(spacy.load('en_core_web_lg'))

    if len(nlp_ref) == 1:
        decoded_text = unidecode.unidecode(text)
        if len(decoded_text) != len(text):
            log.warning('Text must have the same length, falling back to normal text')
            doc = nlp_ref[0](text)
        else:
            doc = nlp_ref[0](decoded_text)
        return [(s.start_char, s.end_char) for s in doc.sents]
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


def add_sentences(questions):
    text_questions = [q['text'] for q in questions]
    sc = create_spark_context()
    sentence_tokenizations = sc.parallelize(text_questions, 4000).map(nlp).collect()
    for q, text, tokenization in zip(questions, text_questions, sentence_tokenizations):
        q['tokenizations'] = tokenization
        # Get the first sentence, end character tokenization
        q['first_sentence'] = text[:tokenization[0][1]]


def questions_to_sqlite(qanta_questions, db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('DROP TABLE IF EXISTS questions;')
    c.execute("""
        CREATE TABLE questions (
          qanta_id INT PRIMARY KEY NOT NULL,
          "text" TEXT NOT NULL, first_sentence TEXT NOT NULL, tokenizations TEXT NOT NULL,
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
            q['qanta_id'], q['text'], q['first_sentence'], str(q['tokenizations']),
            q['answer'], q['page'], q['fold'],
            q['category'], q['subcategory'], q['tournament'], q['difficulty'],
            q['year'], q['proto_id'], q['qdb_id'], q['dataset']
        ) for q in qanta_questions]
    )
    conn.commit()
    conn.close()