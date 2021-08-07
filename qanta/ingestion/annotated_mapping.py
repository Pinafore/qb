from unidecode import unidecode
import pickle
from typing import Dict, List, Optional, Tuple
import os
import yaml
import string
import re
from qanta.util.constants import WIKI_TITLES_PICKLE
from pedroai.result import Ok, Err, Result

PUNCTUATION = string.punctuation
PAREN = re.compile(r"\([^)]*\)")
BRACKET = re.compile(r"\[[^)]*\]")
MULT_SPACE = re.compile(r"\s+")
ANGLE = re.compile(r"<[^>]*>")


def split_and_remove_punc(text):
    for i in text.split():
        word = "".join(x for x in i.lower() if x not in PUNCTUATION)
        if word:
            yield word

# This could be expanded to provide multiple "normalized" answers
def normalize_answer(answer):
    answer = unidecode(answer)
    answer = answer.lower().replace("_ ", " ").replace(" _", " ").replace("_", "")
    answer = answer.replace("{", "").replace("}", "")
    answer = PAREN.sub("", answer)
    answer = BRACKET.sub("", answer)
    answer = ANGLE.sub("", answer)
    answer = MULT_SPACE.sub(" ", answer)
    answer = " ".join(split_and_remove_punc(answer))
    return answer


def load_unambiguous_mapping(path) -> Dict[str, str]:
    """
    Simple dictionary mapping answer strings to Wikipedia Pages
    """
    with open(path) as f:
        return yaml.load(f)["unambiguous"]


def load_ambiguous_mapping(path) -> Dict[str, List[Dict]]:
    """
    For each answer string, it is associated with a list of possible mappings, each represented as a dictionary.
    The keys of the dictionary are "page" which refers to the Wikipedia Page, and "words" which refers to a list of
    strings (words). If any of these words occurs in the question text, then the page should be considered a match
    """
    with open(path) as f:
        return yaml.load(f)["ambiguous"]


ANNOTATED_MAPPING_PATH = "data/internal/page_assignment"
ANNOTATED_MAPPING_FILES = list(string.ascii_lowercase) + ["other"]
QUIZDB_DIRECT = "data/internal/page_assignment/direct/quizdb.yaml"
PROTOBOWL_DIRECT = "data/internal/page_assignment/direct/protobowl.yaml"


class PageAssigner:
    def __init__(self):
        with open(PROTOBOWL_DIRECT) as f:
            self.protobowl_direct = yaml.load(f)["direct"]
        with open(QUIZDB_DIRECT) as f:
            self.quizdb_direct = yaml.load(f)["direct"]

        with open(WIKI_TITLES_PICKLE, "rb") as f:
            self._wiki_titles = pickle.load(f)

        self.ambiguous = {}
        self.unambiguous = {}
        for f in ANNOTATED_MAPPING_FILES:
            path = os.path.join(ANNOTATED_MAPPING_PATH, "ambiguous", f"{f}.yaml")
            a_map = load_ambiguous_mapping(path)
            for k, v in a_map.items():
                self.ambiguous[k] = v

            path = os.path.join(ANNOTATED_MAPPING_PATH, "unambiguous", f"{f}.yaml")
            u_map = load_unambiguous_mapping(path)
            for k, v in u_map.items():
                self.unambiguous[k] = v

    def maybe_ambiguous(self, answer: str, words: List[str]) -> Result[str, str]:
        if answer in self.ambiguous:
            matches = self.ambiguous[answer]
            maybe_page = None
            word_set = set(words)
            for m in matches:
                page_words = m["words"]
                for w in page_words:
                    if w in word_set:
                        if maybe_page is None:
                            maybe_page = m["page"]
                            break
                        else:
                            return Err(
                                f'More than one match found: "{maybe_page}" and "{m["page"]}" for A: "{answer}" W: "'
                                f'{words}"'
                            )
            if maybe_page is None:
                return Err("No match found")
            else:
                return Ok(maybe_page)

        else:
            return Err("No match found")

    def _maybe_assign(
        self, *, answer=None, question_text=None, qdb_id=None, proto_id=None
    ) -> Result[str, str]:
        if answer is None:
            if qdb_id in self.quizdb_direct:
                return Ok(self.quizdb_direct[qdb_id])
            elif proto_id in self.protobowl_direct:
                return Ok(self.protobowl_direct[proto_id])
            else:
                return Err("Cannot have answer, qdb_id, and proto_id all be None")
        else:
            if qdb_id in self.quizdb_direct:
                return Ok(self.quizdb_direct[qdb_id])
            elif proto_id in self.protobowl_direct:
                return Ok(self.protobowl_direct[proto_id])
            else:
                answer = normalize_answer(answer)
                if question_text is None:
                    if answer in self.unambiguous:
                        return Ok(self.unambiguous[answer])
                    else:
                        return Err("No match and no question text")
                else:
                    words = re.sub(r"[^a-zA-Z0-9\s]", "", question_text.lower()).split()
                    maybe_page = self.maybe_ambiguous(answer, words)
                    if maybe_page.is_ok():
                        return maybe_page
                    else:
                        if answer in self.unambiguous:
                            return Ok(self.unambiguous[answer])
                        else:
                            return Err("No match found")

    def maybe_assign(
        self, *, answer=None, question_text=None, qdb_id=None, proto_id=None
    ) -> Tuple[Optional[str], Optional[str]]:
        maybe_page = self._maybe_assign(
            answer=answer, question_text=question_text, qdb_id=qdb_id, proto_id=proto_id
        )
        maybe_page = self._check_page_in_titles(maybe_page)
        if maybe_page.is_ok():
            return maybe_page.ok(), None
        else:
            return None, maybe_page.err()

    def _check_page_in_titles(self, maybe_page: Result[str, str]) -> Result[str, str]:
        if maybe_page.is_err():
            return maybe_page
        else:
            page = maybe_page.ok().replace(" ", "_")
            if page in self._wiki_titles:
                return Ok(page)
            else:
                return Err(f'page="{page}" not in wikipedia titles')
