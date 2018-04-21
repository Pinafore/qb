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
