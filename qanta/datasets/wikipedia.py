from typing import Set
import random
import pickle
from collections import defaultdict
import nltk

from qanta.datasets.abstract import AbstractDataset, TrainingData
from qanta.wikipedia.cached_wikipedia import Wikipedia, extract_wiki_sentences
from qanta.util.io import safe_open


class WikipediaDataset(AbstractDataset):
    def __init__(self, answers: Set[str], n_sentences=5, replace_title_mentions=''):
        super().__init__()
        self.answers = answers
        self.n_sentences = n_sentences
        self.replace_title_mentions = replace_title_mentions

    def training_data(self) -> TrainingData:
        wiki_lookup = Wikipedia()
        wiki_content = []
        wiki_answers = []
        for ans in self.answers:
            if ans not in wiki_lookup:
                continue
            wiki_page = wiki_lookup[ans]
            if len(wiki_page.text) != 0:
                sentences = extract_wiki_sentences(
                    ans, wiki_page.text, self.n_sentences,
                    replace_title_mentions=self.replace_title_mentions
                )
                for sent in sentences:
                    wiki_content.append([sent])
                    wiki_answers.append(ans)

        return wiki_content, wiki_answers, None
