from typing import Set
import nltk
import re
from unidecode import unidecode
from qanta.datasets.abstract import AbstractDataset, TrainingData
from qanta.wikipedia.cached_wikipedia import CachedWikipedia


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
                            content = unidecode(par).lower()

                            # Strip references to the title in a reasonable way
                            ans_words = unidecode(ans).lower().split('_')
                            for w in ans_words:
                                content = content.replace(w, ' ')

                            # Fix up whitespaces
                            content = re.sub('\s+', ' ', content).strip()
                            for sent in nltk.sent_tokenize(content):
                                wiki_content.append([sent])
                                wiki_answers.append(ans)
                        if n_used == self.n_paragraphs:
                            break

        return wiki_content, wiki_answers, None
