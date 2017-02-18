import re
import random
import nltk
from qanta.datasets.abstract import AbstractDataset
from qanta.wikipedia.cached_wikipedia import CachedWikipedia
from qanta.util.environment import QB_WIKI_LOCATION
from qanta.datasets.quiz_bowl import QuizBowlDataset


def is_wiki_header(line):
    if re.match(r'=.*=', line):
        return True
    else:
        return False


def wiki_sentences(page_content):
    lines = [l for l in page_content.split('\n') if l != '' and not is_wiki_header(l)]
    for l in lines:
        for sentence in nltk.sent_tokenize(l):
            yield sentence


class WikipediaDataset(AbstractDataset):
    def __init__(self, min_answers: int, max_sentences: int=40):
        super().__init__()
        self.min_answers = min_answers
        self.max_sentences = max_sentences

    def training_data(self):
        cw = CachedWikipedia(QB_WIKI_LOCATION)
        ds = QuizBowlDataset(2)
        train_data = ds.training_data()
        answer_classes = set(train_data[1])
        train_x = []
        train_y = []

        for page in answer_classes:
            sentences = list(wiki_sentences(cw[page].content))
            sampled_sentences = random.sample(sentences, min(len(sentences), self.max_sentences))
            training_examples = []
            for sentence in sampled_sentences:
                training_examples.append(sentence)
            train_x.append(training_examples)
            train_y.append(page)
        return train_x, train_y
