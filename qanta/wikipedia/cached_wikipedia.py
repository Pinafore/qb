from typing import Dict
import os
import json
import csv
import pickle
import re
from collections import namedtuple
import nltk
from unidecode import unidecode

from qanta import qlogging
from qanta.datasets.quiz_bowl import QantaDatabase
from qanta.util.constants import (
    COUNTRY_LIST_PATH,
    WIKI_DUMP_REDIRECT_PICKLE,
    WIKI_LOOKUP_PATH,
)


log = qlogging.get(__name__)

COUNTRY_SUB = ["History_of_", "Geography_of_"]

WikipediaPage = namedtuple("WikipediaPage", ["id", "title", "text", "url"])


def normalize_wikipedia_title(title):
    return title.replace(" ", "_")


def create_wikipedia_title_pickle(dump_path, disambiguation_pages_path, output_path):
    from qanta.spark import create_spark_session

    with open(disambiguation_pages_path) as f:
        disambiguation_pages = set(json.load(f))

    spark = create_spark_session()
    wiki_df = spark.read.json(dump_path)
    rows = wiki_df.select("title", "id").distinct().collect()
    content_pages = [r for r in rows if int(r.id) not in disambiguation_pages]
    clean_titles = {normalize_wikipedia_title(r.title) for r in content_pages}

    with open(output_path, "wb") as f:
        pickle.dump(clean_titles, f)
    spark.stop()


def create_wikipedia_cache(
    parsed_wiki_path="data/external/wikipedia/parsed-wiki", output_path=WIKI_LOOKUP_PATH
):
    from qanta.spark import create_spark_context

    sc = create_spark_context()
    db = QantaDatabase()
    train_questions = db.train_questions
    answers = {q.page for q in train_questions}
    b_answers = sc.broadcast(answers)
    # Paths used in spark need to be absolute and it needs to exist
    page_path = os.path.abspath(parsed_wiki_path)
    page_pattern = os.path.join(page_path, "*", "*")

    def parse_page(json_text):
        page = json.loads(json_text)
        return {
            "id": int(page["id"]),
            "title": page["title"].replace(" ", "_"),
            "text": page["text"],
            "url": page["url"],
        }

    wiki_pages = (
        sc.textFile(page_pattern)
        .map(parse_page)
        .filter(lambda p: p["title"] in b_answers.value)
        .collect()
    )
    wiki_lookup = {p["title"]: p for p in wiki_pages}
    with open(output_path, "w") as f:
        json.dump(wiki_lookup, f)

    return wiki_lookup


def create_wikipedia_redirect_pickle(redirect_csv, output_pickle):
    countries = {}
    with open(COUNTRY_LIST_PATH) as f:
        for line in f:
            k, v = line.split("\t")
            countries[k] = v.strip()

    db = QantaDatabase()
    pages = {q.page for q in db.train_questions}

    with open(redirect_csv) as redirect_f:
        redirects = {}
        n_total = 0
        n_selected = 0
        for row in csv.reader(redirect_f, quotechar='"', escapechar="\\"):
            n_total += 1
            source = row[0]
            target = row[1]
            if (
                target not in pages
                or source in countries
                or target.startswith("WikiProject")
                or target.endswith("_topics")
                or target.endswith("_(overview)")
            ):
                continue
            else:
                redirects[source] = target
                n_selected += 1

        log.info(
            "Filtered {} raw wikipedia redirects to {} matching redirects".format(
                n_total, n_selected
            )
        )

    with open(output_pickle, "wb") as output_f:
        pickle.dump(redirects, output_f)


def extract_wiki_sentences(title, text, n_sentences, replace_title_mentions=""):
    """
    Extracts the first n_paragraphs from the text of a wikipedia page corresponding to the title.
    strip_title_mentions and replace_title_mentions control handling of references to the title in text.
    Oftentimes QA models learn *not* to answer entities mentioned in the question so this helps deal with this
    in the domain adaptation case.

    :param title: title of page
    :param text: text of page
    :param n_paragraphs: number of paragraphs to use
    :param replace_title_mentions: Replace mentions with the provided string token, by default removing them
    :return:
    """
    # Get simplest representation of title and text
    title = unidecode(title).replace("_", " ")
    text = unidecode(text)

    # Split on non-alphanumeric
    title_words = re.split("[^a-zA-Z0-9]", title)
    title_word_pattern = "|".join(re.escape(w.lower()) for w in title_words)

    # Breaking by newline yields paragraphs. Ignore the first since its always just the title
    paragraphs = [p for p in text.split("\n") if len(p) != 0][1:]
    sentences = []
    for p in paragraphs:
        formatted_text = re.sub(
            title_word_pattern, replace_title_mentions, p, flags=re.IGNORECASE
        )
        # Cleanup whitespace
        formatted_text = re.sub("\s+", " ", formatted_text).strip()

        sentences.extend(nltk.sent_tokenize(formatted_text))

    return sentences[:n_sentences]


class Wikipedia:
    def __init__(
        self, lookup_path=WIKI_LOOKUP_PATH, dump_redirect_path=WIKI_DUMP_REDIRECT_PICKLE
    ):
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
        self.countries = {}
        self.redirects = {}
        self.lookup_path = lookup_path
        self.dump_redirect_path = dump_redirect_path
        with open(lookup_path, "rb") as f:
            raw_lookup: Dict[str, Dict] = json.load(f)
            self.lookup: Dict[str, WikipediaPage] = {
                title: WikipediaPage(
                    page["id"], page["title"], page["text"], page["url"]
                )
                for title, page in raw_lookup.items()
            }

        if COUNTRY_LIST_PATH:
            with open(COUNTRY_LIST_PATH) as f:
                for line in f:
                    k, v = line.split("\t")
                    self.countries[k] = v.replace(" ", "_").strip()

        if os.path.exists(self.dump_redirect_path):
            with open(self.dump_redirect_path, "rb") as f:
                self.redirects = pickle.load(f)
        else:
            raise ValueError(
                f"{self.dump_redirect_path} missing, run: luigi --module qanta.pipeline.preprocess "
                f"WikipediaRedirectPickle"
            )

    def load_country(self, key: str):
        content = self.lookup[key]
        for page in [f"{prefix}{self.countries[key]}" for prefix in COUNTRY_SUB]:
            if page in self.lookup:
                content = content + " " + self.lookup[page].text
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
