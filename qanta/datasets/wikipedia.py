from typing import Set
import random
import pickle
from collections import defaultdict
import nltk

from qanta.datasets.abstract import AbstractDataset, TrainingData
from qanta.wikipedia.cached_wikipedia import Wikipedia, extract_wiki_sentences
from qanta.tagme import TagmeClient
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
        wiki_lookup = Wikipedia()

        page_sentences = defaultdict(list)
        for ans in answers:
            if ans not in wiki_lookup:
                continue
            wiki_page = wiki_lookup[ans]
            if len(wiki_page.text) != 0:
                sentences = nltk.sent_tokenize(wiki_page.text)
                random.shuffle(sentences)
                clean_sentences, all_mentions = client.tag_mentions(sentences)
                for sent, mentions in zip(clean_sentences, all_mentions):
                    page_mentions = {m.page for m in mentions}
                    n_mentions = len(page_mentions)
                    for page in page_mentions.intersection(answers):
                        raise NotImplementedError('Need to fix this to use extract_wiki_sentences')
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
            sorted_sentences = sorted(page_sentences[ans], reverse=True, key=lambda x: x[0])
            sent_list = [t[1] for t in sorted_sentences]

            for sentence in sent_list[:self.n_examples]:
                tagme_content.append([sentence])
                tagme_answers.append(ans)

        return tagme_content, tagme_answers, None
