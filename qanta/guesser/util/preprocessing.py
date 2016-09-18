import re
import regex

_ftp = ["for 10 points, ", "for 10 points--", "for ten points, ", "for 10 points ",
        "for ten points ", "ftp,", "ftp"]


def preprocess_text(q, ners=None):
    if ners is None:
        ners = []
    q = q.strip().lower()

    # remove pronunciation guides and other formatting extras
    q = q.replace(' (*) ', ' ')
    q = q.replace('\n', '')
    q = q.replace('mt. ', 'mt ')
    q = q.replace(', for 10 points, ', ' ')
    q = q.replace(', for ten points, ', ' ')
    q = q.replace('--for 10 points--', ' ')
    q = q.replace(', ftp, ', ' ')
    q = q.replace('{', '')
    q = q.replace('}', '')
    q = q.replace('~', '')
    q = q.replace('(*)', '')
    q = q.replace('*', '')
    q = re.sub(r'\[.*?\]', '', q)
    q = re.sub(r'\(.*?\)', '', q)

    for phrase in _ftp:
        q = q.replace(phrase, ' ')

    # remove punctuation
    q = regex.sub(r"\p{P}+", " ", q)

    q = regex.sub(' +', ' ', q)

    # simple ner (replace answers w/ concatenated versions)
    for ner in ners:
        q = q.replace(ner, ner.replace(' ', '_'))

    return q
