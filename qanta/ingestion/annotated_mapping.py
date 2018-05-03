from unidecode import unidecode
from collections import defaultdict, Counter
import string
import re

PUNCTUATION = string.punctuation
PAREN = re.compile(r'\([^)]*\)')
BRACKET = re.compile(r'\[[^)]*\]')
MULT_SPACE = re.compile(r'\s+')
ANGLE = re.compile(r'<[^>]*>')


def split_and_remove_punc(text):
    for i in text.split():
        word = "".join(x for x in i.lower() if x not in PUNCTUATION)
        if word:
            yield word


def normalize_answer(answer):
    answer = answer.lower().replace("_ ", " ").replace(" _", " ").replace("_", "")
    answer = answer.replace("{", "").replace("}", "")
    answer = PAREN.sub('', answer)
    answer = BRACKET.sub('', answer)
    answer = ANGLE.sub('', answer)
    answer = MULT_SPACE.sub(' ', answer)
    answer = " ".join(split_and_remove_punc(answer))
    return answer


def parse_unambiguous_mappings(path):
    with open(path) as f:
        mappings = {}
        for line in f:
            splits = line.strip().split('\t')
            if len(splits) != 2:
                continue
            else:
                source, target = splits
                mappings[source] = target

    return mappings
