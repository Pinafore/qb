import sqlite3
import spacy
import unidecode
from qanta import qlogging
from qanta.spark import create_spark_context


log = qlogging.get(__name__)

# Each spark worker needs to load its own copy of the NLP model
# separately since its not serializable and thus not broadcastable
nlp_ref = []
AVG_WORD_LENGTH = 5
MIN_WORDS = 12
MIN_CHAR_LENGTH = AVG_WORD_LENGTH * MIN_WORDS


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
        tokenizations = [(s.start_char, s.end_char) for s in doc.sents]
        first_end_pos = None
        if len(tokenizations) == 0:
            raise ValueError('Zero length question with respect to sentences not allowed')

        for start, end in tokenizations:
            if end < MIN_CHAR_LENGTH:
                continue
            else:
                first_end_pos = end
                break

        if first_end_pos is None:
            first_end_pos = tokenizations[-1][1]

        final_tokenizations = [(0, first_end_pos)]
        for start, end in tokenizations:
            if end <= first_end_pos:
                continue
            else:
                final_tokenizations.append((start, end))

        return final_tokenizations
    else:
        raise ValueError('There should be exactly one nlp model per spark worker')


def format_qanta_json(questions, version):
    return {
        'questions': questions,
        'version': version,
        'maintainer_name': 'Pedro Rodriguez',
        'maintainer_contact': 'entilzha@umiacs.umd.edu',
        'maintainer_website': 'http://pedrorodriguez.io',
        'project_website': 'https://github.com/pinafore/qb'
    }


def add_sentences_(questions):
    text_questions = [q['text'] for q in questions]
    sc = create_spark_context()
    sentence_tokenizations = sc.parallelize(text_questions, 4000).map(nlp).collect()
    for q, text, tokenization in zip(questions, text_questions, sentence_tokenizations):
        q['tokenizations'] = tokenization
        # Get the 0th sentence, end character tokenization (tuple position 1)
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
          fold TEXT NOT NULL, gameplay BOOLEAN,
          category TEXT, subcategory TEXT,
          tournament TEXT, difficulty TEXT, year INT,
          proto_id INT, qdb_id INT, dataset TEXT NOT NULL
        )
    """)
    c.executemany(
        'INSERT INTO questions values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
        [(
            q['qanta_id'], q['text'], q['first_sentence'], str(q['tokenizations']),
            q['answer'], q['page'], q['fold'], q['gameplay'],
            q['category'], q['subcategory'], q['tournament'], q['difficulty'],
            q['year'], q['proto_id'], q['qdb_id'], q['dataset']
        ) for q in qanta_questions]
    )
    conn.commit()
    conn.close()