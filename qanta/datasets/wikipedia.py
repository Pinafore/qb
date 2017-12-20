from typing import Set
import random
import pickle
from collections import defaultdict
import nltk
import re
from unidecode import unidecode

from qanta.datasets.abstract import AbstractDataset, TrainingData
from qanta.wikipedia.cached_wikipedia import CachedWikipedia
from qanta.tagme import TagmeClient
from qanta.util.io import safe_open


def strip_title_references(title, text):
    """
    Search the text for references to words in the title and remove them
    """
    text = unidecode(text).lower()
    title_words = unidecode(title).lower().split('_')
    for w in title_words:
        text = text.replace(w, ' ')

    # Fix up whitespaces
    return re.sub('\s+', ' ', text).strip()

class WikipediaDataset(AbstractDataset):
    def __init__(self, answers: Set[str], n_paragraphs=3):
        super().__init__()
        self.answers = answers
        self.n_paragraphs = n_paragraphs

    def training_data(self) -> TrainingData:
        cw = CachedWikipedia()
        wiki_content = []
        wiki_answers = []
        for ans in self.answers:
            wiki_page = cw[ans]
            if len(wiki_page.content) != 0:
                # Take the first paragraph, skipping the initial title and empty line after
                paragraphs = wiki_page.content.split('\n')
                if len(paragraphs) > 2:
                    n_used = 0
                    for par in paragraphs[2:]:
                        if len(par) != 0:
                            n_used += 1
                            content = strip_title_references(ans, par)
                            for sent in nltk.sent_tokenize(content):
                                wiki_content.append([sent])
                                wiki_answers.append(ans)
                        if n_used == self.n_paragraphs:
                            break

        return wiki_content, wiki_answers, None


class TagmeWikipediaDataset(AbstractDataset):
    def __init__(self, location='output/tagme/tagme-wikipedia.pickle', n_examples=20):
        """
        :param answers: Answer set to use in QB normalized format
        :param n_examples: Number of examples per answer to add
        """
        super().__init__()
        self.n_examples = n_examples
        self.location = location

    def build(self, answers: Set[str], save=True):
        client = TagmeClient()
        cw = CachedWikipedia()

        page_sentences = defaultdict(list)
        for ans in answers:
            wiki_page = cw[ans]
            if len(wiki_page.content) != 0:
                sentences = nltk.sent_tokenize(wiki_page.content)
                random.shuffle(sentences)
                clean_sentences, all_mentions = client.tag_mentions(sentences)
                for sent, mentions in zip(clean_sentences, all_mentions):
                    page_mentions = {m.page for m in mentions}
                    n_mentions = len(page_mentions)
                    for page in page_mentions.intersection(answers):
                        stripped_sent = strip_title_references(page, sent)
                        page_sentences[page].append((n_mentions, stripped_sent))

        if save:
            with safe_open(self.location, 'wb') as f:
                pickle.dump(page_sentences, f)

        return page_sentences


    def training_data(self):
        with open(self.location, 'rb') as f:
            page_sentences = pickle.load(f)

        tagme_content = []
        tagme_answers = []

        for ans in page_sentences:
            n_mentions, sent_list = sorted(page_sentences[ans], reverse=True, key=lambda x: x[0])

            for sentence in sent_list[:self.n_examples]:
                tagme_content.append([sentence])
                tagme_answers.append(ans)

        return tagme_content, tagme_answers, None
