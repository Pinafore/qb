from typing import Dict
import os
import json
import csv
import pickle
from collections import namedtuple

from qanta import qlogging
from qanta.datasets.quiz_bowl import QuestionDatabase
from qanta.util.constants import COUNTRY_LIST_PATH, WIKI_DUMP_REDIRECT_PICKLE, WIKI_LOOKUP_PATH


log = qlogging.get(__name__)

COUNTRY_SUB = ["History_of_", "Geography_of_"]

WikipediaPage = namedtuple('WikipediaPage', ['id', 'title', 'text', 'url'])


def normalize_wikipedia_title(title):
    return title.replace(' ', '_')


def create_wikipedia_title_pickle(dump_path, output_path):
    from qanta.spark import create_spark_session

    spark = create_spark_session()
    wiki_df = spark.read.json(dump_path)
    raw_titles = wiki_df.select('title').distinct().collect()
    clean_titles = {normalize_wikipedia_title(r.title) for r in raw_titles}
    with open(output_path, 'wb') as f:
        pickle.dump(clean_titles, f)
    spark.stop()


def create_wikipedia_cache(parsed_wiki_path='data/external/wikipedia/parsed-wiki', output_path=WIKI_LOOKUP_PATH):
    from qanta.spark import create_spark_context

    sc = create_spark_context()
    db = QuestionDatabase()
    questions = list(db.all_questions().values())
    train_questions = [q for q in questions if q.fold == 'guesstrain' or q.fold == 'buzzertrain']
    answers = {q.page for q in train_questions}
    b_answers = sc.broadcast(answers)
    # Paths used in spark need to be absolute and it needs to exist
    page_path = os.path.abspath(parsed_wiki_path)
    page_pattern = os.path.join(page_path, '*', '*')

    def parse_page(json_text):
        page = json.loads(json_text)
        return WikipediaPage(int(page['id']), page['title'].replace(' ', '_'), page['text'], page['url'])

    wiki_pages = sc.textFile(page_pattern).map(parse_page).filter(lambda p: p.title in b_answers.value).collect()
    wiki_lookup = {p.title: p for p in wiki_pages}
    with open(output_path, 'wb') as f:
        pickle.dump(wiki_lookup, f)

    return wiki_lookup


def create_wikipedia_redirect_pickle(redirect_csv, output_pickle):
    countries = {}
    with open(COUNTRY_LIST_PATH) as f:
        for line in f:
            k, v = line.split('\t')
            countries[k] = v.strip()

    db = QuestionDatabase()
    pages = set(db.all_answers().values())

    with open(redirect_csv) as redirect_f:
        redirects = {}
        n_total = 0
        n_selected = 0
        for row in csv.reader(redirect_f, quotechar='"', escapechar='\\'):
            n_total += 1
            source = row[0]
            target = row[1]
            if (target not in pages or source in countries or
                    target.startswith('WikiProject') or
                    target.endswith("_topics") or
                    target.endswith("_(overview)")):
                continue
            else:
                redirects[source] = target
                n_selected += 1

        log.info('Filtered {} raw wikipedia redirects to {} matching redirects'.format(n_total, n_selected))

    with open(output_pickle, 'wb') as output_f:
        pickle.dump(redirects, output_f)


class Wikipedia:
    def __init__(self, lookup_path=WIKI_LOOKUP_PATH):
        """
        CachedWikipedia provides a unified way and easy way to access Wikipedia pages. Its design is motivated by:
        1) Getting a wikipedia page should function as a simple python dictionary access
        2) It should support access to pages using non-canonical names by resolving them to canonical names

        The following sections explain how the different levels of caching work as well as how redirects work

        Redirects
        To support some flexibility in requesting pages that are very close matches we have two sources of redirects.
        The first is based on wikipedia database dumps which is the most reliable.  On top
        of this we do the very light preprocessing step of replacing whitespace with underscores since the canonical
        page names in the wikipedia database dumps contains an underscore instead of whitespace (a difference from the
        HTTP package which defaults to the opposite)
        """
        self.dump_redirect_path = WIKI_DUMP_REDIRECT_PICKLE
        self.countries = {}
        self.redirects = {}
        self.lookup_path = lookup_path
        with open(lookup_path, 'rb') as f:
            self.lookup: Dict[str, WikipediaPage] = pickle.load(f)

        if COUNTRY_LIST_PATH:
            with open(COUNTRY_LIST_PATH) as f:
                for line in f:
                    k, v = line.split('\t')
                    self.countries[k] = v.replace(' ', '_').strip()

        if os.path.exists(self.dump_redirect_path):
            with open(self.dump_redirect_path, 'rb') as f:
                self.redirects = pickle.load(f)
        else:
            raise ValueError(
                'The redirect file (%s) from the dump is missing, run: luigi --module qanta.pipeline.preprocess WikipediaRedirectPickle' % self.dump_redirect_path)

    def load_country(self, key: str):
        content = self[key]
        for page in [f"{prefix}{self.countries[key]}" for prefix in COUNTRY_SUB]:
            if page in self.lookup:
                content = content + ' ' + self.lookup[page].text
        return content

    def __getitem__(self, key: str) -> WikipediaPage:
        if key in self.countries:
            return self.load_country(key)
        else:
            return self.lookup[key]

    def __contains__(self, item):
        return item in self.lookup

    def __len__(self):
        return len(self.lookup)
