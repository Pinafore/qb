from typing import Tuple, Set, Dict, List, Callable, Iterable, Optional
from pprint import pformat
import csv
import json
import re
import pickle
from collections import defaultdict, Counter, OrderedDict
from unidecode import unidecode
import tqdm

from nltk.corpus import wordnet

from qanta import qlogging
from qanta.util.constants import (
    WIKI_TITLES_PICKLE,
    ALL_WIKI_REDIRECTS,
    GUESSER_TRAIN_FOLD,
    BUZZER_TRAIN_FOLD,
)
from qanta.util.io import safe_open
from qanta.ingestion.annotated_mapping import PageAssigner


log = qlogging.get(__name__)

ExpansionRule = Callable[[str], Iterable[str]]
MatchRule = Callable[[str], Optional[str]]


def make_source_list(
    wiki_titles: Set[str], wiki_redirects_source: Dict[str, str]
) -> List[Tuple[str, Dict[str, str], bool]]:
    exact_titles = {}
    unicode_titles = {}
    lower_titles = {}
    lower_unicode_titles = {}
    for title in wiki_titles:
        # exact titles
        exact_titles[title] = title

        # unicode titles
        u_title = unidecode(title)
        if len(title) != 1 or len(title) == len(u_title):
            unicode_titles[u_title] = title

        # lower titles
        l_title = title.lower()
        lower_titles[l_title] = title

        u_l_title = unidecode(l_title)
        if len(title) != 1 or len(title) == len(u_l_title):
            lower_unicode_titles[u_l_title] = title

    exact_wiki_redirects = {}
    unicode_wiki_redirects = {}
    lower_wiki_redirects = {}
    lower_unicode_wiki_redirects = {}
    for text, page in wiki_redirects_source.items():
        exact_wiki_redirects[text] = page

        u_text = unidecode(text)
        if len(text) != 1 or len(text) == len(u_text):
            unicode_wiki_redirects[u_text] = page

        l_text = text.lower()
        lower_wiki_redirects[l_text] = page

        u_l_text = unidecode(l_text)
        if len(text) != 1 or len(text) == len(u_l_text):
            lower_unicode_wiki_redirects[u_l_text] = page

    return [
        ("exact_title", exact_titles, False),
        ("exact_redirect", exact_wiki_redirects, False),
        ("unicode_title", unicode_titles, False),
        ("lower_title", lower_titles, True),
        ("lower_redirect", lower_wiki_redirects, True),
        ("lower_unicode_title", lower_unicode_titles, True),
    ]


DISAMBIG_SYNONYMS = {
    "mythology": {"myth", "deity", "god", "goddess"},
    "mathematics": {"math"},
}
DISAMB_KEYWORD_BLACKLIST = {"name", "disambiguation"}

# List of exact title with keyword indicating when the title should be used
# For example, [('Paris_(mythology)', 'mythology'), ('Paris_(band)', 'band')]
AmbigOptions = Set[Tuple[str, str]]
# Answer string to AmbigOptions
# Following the above example (referred to as paris_options, this could be {'Paris': paris_options}
AmbigMap = Dict[str, AmbigOptions]


def make_disamb_list(wiki_titles: Set[str]) -> AmbigMap:
    disamb_candidates = defaultdict(set)
    disamb_re = re.compile(r"(^[^\(\)]+)\_\((.+)\)$")
    for t in wiki_titles:
        m = re.match(disamb_re, t)
        if m is not None:
            amb_title, keyword = m.group(1), m.group(2)
            if keyword not in DISAMB_KEYWORD_BLACKLIST:
                if "_" in keyword:
                    if "film" in keyword:
                        disamb_candidates[amb_title].add((t, "film"))
                else:
                    disamb_candidates[amb_title].add((t, keyword))

    return disamb_candidates


def mapping_rules_to_answer_map(
    expansion_rules: List[Tuple[str, int, ExpansionRule]],
    match_rules: List[Tuple[str, int, MatchRule]],
    wiki_titles: Set[str],
    wiki_redirects_source,
    unmapped_answers: Set[str],
):
    log.info("Creating wikipedia title variants for matching")

    source_list = make_source_list(wiki_titles, wiki_redirects_source)
    disamb_candidates = make_disamb_list(wiki_titles)
    answer_map = {}
    amb_answer_map = defaultdict(list)

    # Clone the set to prevent accidental mutation of the original
    unmapped_answers = set(unmapped_answers)

    n_unmapped = len(unmapped_answers)
    log.info(f"{n_unmapped} Unmapped Answers Exist\nStarting Answer Mapping\n")
    expansion_answer_map = defaultdict(
        dict
    )  # type: Dict[str, Dict[str, Tuple[int, str]]]
    for name, priority, rule_func in sorted(
        expansion_rules, key=lambda x: x[1], reverse=True
    ):
        log.info(f"Applying expansion rule: {name}")
        for raw_ans in unmapped_answers:
            for exp_ans in rule_func(raw_ans):
                exp_ans = exp_ans.strip()
                if exp_ans in expansion_answer_map[raw_ans]:
                    curr_priority, _ = expansion_answer_map[raw_ans][exp_ans]
                    if priority > curr_priority:
                        expansion_answer_map[raw_ans][exp_ans] = priority, name
                else:
                    expansion_answer_map[raw_ans][exp_ans] = priority, name

    sorted_match_rules = sorted(match_rules, key=lambda x: x[1], reverse=True)
    report = {"expansion": {}, "match": {}, "source": {}}
    for original_ans, ans_expansions in tqdm.tqdm(expansion_answer_map.items()):
        # We don't need the expansion priority anymore, its already been sorted
        for match_name, _, rule_func in sorted_match_rules:
            # We don't need the match priority anymore, its already been sorted
            for raw_ans, (_, expansion_name) in sorted(
                ans_expansions.items(), key=lambda x: x[1], reverse=True
            ):
                rule_ans = re.sub(r"\s+", " ", rule_func(raw_ans)).strip()
                lower_ans = rule_ans.lower()
                is_upper = rule_ans.isupper()

                # If we already have an answer, by definition it must be of higher priority so we do not allow
                # overwriting it.
                rule_ans = re.sub(r"\s+", " ", rule_func(raw_ans)).strip()
                if original_ans not in answer_map:
                    match, source = find_match(
                        rule_ans, lower_ans, is_upper, source_list
                    )

                    if match is not None:
                        answer_map[original_ans] = match
                        report["expansion"][original_ans] = match_name
                        report["match"][original_ans] = expansion_name
                        report["source"][original_ans] = source

                amb_match, ambig_source = find_amb_match(rule_ans, disamb_candidates)
                if amb_match is not None:
                    amb_answer_map[original_ans].extend(amb_match)

    # Ambig options should be unique, but respect insertion order. They must also be json serializable
    for k in amb_answer_map:
        options = amb_answer_map[k]
        unique_options = OrderedDict()
        for o in options:
            unique_options[o] = None
        amb_answer_map[k] = list(unique_options.keys())

    n_mapped = len(answer_map)
    expansion_counts = Counter(report["expansion"].values())
    match_counts = Counter(report["match"].values())
    source_counts = Counter(report["source"].values())
    report["expansion_counts"] = expansion_counts
    report["match_counts"] = match_counts
    report["source_counts"] = source_counts
    log.info(f"Expansion Breakdown:\n{pformat(expansion_counts)}")
    log.info(f"Match Breakdown:\n{pformat(match_counts)}")
    log.info(f"Source Breakdown:\n{pformat(source_counts)}")
    log.info(
        f"\nAnswer Mapping Complete\n{n_unmapped - n_mapped} Unmapped Remain, {n_mapped} Mappings Found"
    )

    return answer_map, amb_answer_map, unmapped_answers, report


def try_match(ans_text, title_map):
    und_text = ans_text.replace(" ", "_")
    if ans_text in title_map:
        return title_map[ans_text]
    elif und_text in title_map:
        return title_map[und_text]
    else:
        return None


def find_match(
    rule_ans, lower_ans, is_upper, source_list
) -> Tuple[Optional[str], Optional[str]]:
    for source_name, source, lower in source_list:
        if lower:
            if not is_upper:
                m = try_match(lower_ans, source)
                if m is not None:
                    return m, source_name
        else:
            m = try_match(rule_ans, source)
            if m is not None:
                return m, source_name
    return None, None


def find_amb_match(
    rule_ans, source: AmbigMap
) -> Tuple[Optional[AmbigOptions], Optional[str]]:
    m = try_match(rule_ans, source)
    if m is not None:
        return m, "disambig_exact"
    else:
        return None, None


# Expansion rule functions
def or_rule(ans):
    splits = re.split("[^a-zA-Z]+or[^a-zA-Z]+", ans)
    if len(splits) > 1:
        formatted_splits = [s.strip() for s in splits]
        return formatted_splits
    else:
        return ()


def exact_rule(ans):
    return (ans,)


def prompt_rule(ans):
    l_ans = ans.lower()
    if "accept" in l_ans or "prompt" in l_ans or "pronounce" in l_ans:
        m = re.match(
            r"(.+)\(.*(?:accept|prompt|pronounce).*\)", ans, flags=re.IGNORECASE
        )
        if m is not None:
            return (m.group(1).strip(),)

        m = re.match(
            r"(.+)\[.*(?:accept|prompt|pronounce).*\]", ans, flags=re.IGNORECASE
        )
        if m is not None:
            return (m.group(1).strip(),)

        return ()
    elif "or" in l_ans:
        m = re.match(r"(.+)\(.*(?:or).*\)", ans, flags=re.IGNORECASE)
        if m is not None:
            return (m.group(1).strip(),)

        m = re.match(r"(.+)\[.*(?:or).*\]", ans, flags=re.IGNORECASE)
        if m is not None:
            return (m.group(1).strip(),)

        return ()
    else:
        return ()


def the_rule(ans):
    l_ans = ans.lower()
    if "the " in l_ans:
        return (re.sub("the ", "", ans, flags=re.IGNORECASE),)
    else:
        return ("the " + ans, "The " + ans)


def plural_rule(ans):
    singular = wordnet.morphy(ans)
    if singular is not None and singular != ans:
        return singular
    else:
        return ans


def apostraphe_rule(ans):
    if "’" in ans:
        return (ans.replace("’", "'"), ans.replace("’", ""))
    else:
        return ()


def answer_rule(ans):
    l_ans = ans.lower()
    if "answers:" in l_ans:
        return (re.sub(r"answers:", "", ans, flags=re.IGNORECASE),)
    elif "answer:" in l_ans:
        return (re.sub(r"answer:", "", ans, flags=re.IGNORECASE),)
    else:
        return ()


def optional_text_rule(ans):
    candidate = re.sub(r"\(.+?\)", "", ans)
    if candidate != ans:
        return (candidate,)
    else:
        return ()


def parens_rule(ans):
    if "(" in ans and ")" in ans:
        return (ans.replace("(", "").replace(")", ""),)
    else:
        return ()


def sir_rule(ans):
    if "sir" in ans.lower():
        return (re.sub(r"sir", "", ans, flags=re.IGNORECASE),)
    else:
        return ()


def amp_rule(ans):
    if "&" in ans:
        return (ans.replace("&", "and"),)
    else:
        return ()


def unicode_rule(ans):
    unicode_ans = unidecode(ans)
    if ans != unicode_ans:
        # Handle things like Korean characters redirecting to Hannibal. unidecode expands these to multi-character
        # sequences which are not words. We should ignore these
        if len(ans) == 1 and len(ans) != len(unicode_ans):
            return ()
        else:
            return (unicode_ans,)
    else:
        return ()


# Match Rule Functions
def remove_braces(text):
    return re.sub(r"[{}]", "", text)


def remove_quotes(text):
    return re.sub(r'["“”]', "", text)


def remove_parens(text):
    return re.sub(r"[\(\)]", "", text)


def compose(*funcs):
    def composed_function(x):
        for f in funcs:
            x = f(x)
        return x

    return composed_function


def create_expansion_rules() -> List[Tuple[str, int, ExpansionRule]]:
    # Apply this rules to generate multiple possible answers from one distinct answer
    expansion_rules = [
        ("exact", 100, exact_rule),
        ("unicode", 50, unicode_rule),
        ("optional-text", 40, optional_text_rule),
        ("apostraphe", 30, apostraphe_rule),
        ("amp", 27, amp_rule),
        ("parens", 25, parens_rule),
        ("answer", 20, answer_rule),
        ("the", 10, the_rule),
        ("sir", 5, sir_rule),
        ("or", 1, or_rule),
        ("prompt", 0, prompt_rule),
    ]
    return expansion_rules


def create_match_rules() -> List[Tuple[str, int, MatchRule]]:
    # Take an answer, format it, then check if there is an exact match
    match_rules = [
        ("exact match", 10, lambda x: x),
        ("braces", 1, remove_braces),
        ("quotes", 1, remove_quotes),
        ("parens", 1, remove_parens),
        ("braces+quotes", 1, compose(remove_braces, remove_quotes)),
        ("braces+plural", 0, compose(remove_braces, plural_rule)),
        ("quotes+braces+plural", 0, compose(remove_braces, remove_quotes, plural_rule)),
    ]
    return match_rules


def create_answer_map(unmapped_qanta_questions):
    expansion_rules = create_expansion_rules()
    match_rules = create_match_rules()

    log.info("Loading questions")
    raw_unmapped_answers = {q["answer"] for q in unmapped_qanta_questions}
    unmapped_lookup = defaultdict(list)
    for q in unmapped_qanta_questions:
        unmapped_lookup[q["answer"]].append(q)

    log.info("Loading wikipedia titles")
    wiki_titles = read_wiki_titles()

    wiki_redirect_map = read_wiki_redirects(wiki_titles)

    log.info("Starting Answer Mapping Process")
    answer_map, amb_answer_map, unbound_answers, report = mapping_rules_to_answer_map(
        expansion_rules,
        match_rules,
        wiki_titles,
        wiki_redirect_map,
        raw_unmapped_answers,
    )
    return answer_map, amb_answer_map, unbound_answers, report


def write_answer_map(
    answer_map, amb_answer_map, unbound_answers, answer_map_path, unbound_answer_path
):
    with safe_open(answer_map_path, "w") as f:
        json.dump({"answer_map": answer_map, "ambig_answer_map": amb_answer_map}, f)

    with safe_open(unbound_answer_path, "w") as f:
        json.dump({"unbound_answers": list(sorted(unbound_answers))}, f)


def unmapped_to_mapped_questions(
    unmapped_qanta_questions,
    answer_map,
    ambig_answer_map,
    unmappable,
    page_assigner: PageAssigner,
):
    proto_unmappable = set(unmappable["proto"])
    qdb_unmappable = set(unmappable["quizdb"])
    train_unmatched_questions = []
    test_unmatched_questions = []
    match_report = {}
    for q in unmapped_qanta_questions:
        answer = q["answer"]
        qanta_id = int(q["qanta_id"])
        proto_id = q["proto_id"]
        qdb_id = q["qdb_id"]
        fold = q["fold"]

        if proto_id in proto_unmappable or qdb_id in qdb_unmappable:
            match_report[qanta_id] = {
                "result": "none",
                "annotated_error": None,
                "automatic_error": "Unmappable answer",
                "annotated_page": None,
                "automatic_page": None,
            }
            if fold == GUESSER_TRAIN_FOLD or fold == BUZZER_TRAIN_FOLD:
                train_unmatched_questions.append(q)
            else:
                test_unmatched_questions.append(q)
            continue

        annotated_page, annotated_error = page_assigner.maybe_assign(
            answer=answer, question_text=q["text"], qdb_id=qdb_id, proto_id=proto_id
        )
        automatic_page = answer_map[answer] if answer in answer_map else None
        ambig_automatic_error = None
        ambig_automatic_page = None
        if answer in ambig_answer_map:
            words = set(q["text"].lower().split())
            options = ambig_answer_map[answer]
            ambig_automatic_page = None
            for page, keyword in options:
                if keyword in words:
                    if ambig_automatic_page is None and ambig_automatic_error is None:
                        ambig_automatic_page = page
                    else:
                        if ambig_automatic_error is None:
                            ambig_automatic_page = None
                            ambig_automatic_error = "Ambig Matches: " + page
                        else:
                            ambig_automatic_page = None
                            ambig_automatic_error += " " + page

        automatic_error = None
        if automatic_page is not None and ambig_automatic_page is not None:
            if automatic_page != ambig_automatic_page:
                # Use automatic_page, but emit a warning
                automatic_error = (
                    f"Ambiguity Warning: {automatic_page} {ambig_automatic_page}"
                )
        elif automatic_page is None and ambig_automatic_page is not None:
            # This is the safe case where we attempt to match when there isn't a match already
            automatic_page = ambig_automatic_page
        elif automatic_page is not None and ambig_automatic_page is None:
            # Do nothing here since by default automatic_page is used
            pass
        else:
            # if both are None, then there is no automatic match
            automatic_error = "No match"

        if (annotated_page is None) and (automatic_page is None):
            match_report[qanta_id] = {
                "result": "none",
                "annotated_error": annotated_error,
                "automatic_error": automatic_error,
                "annotated_page": annotated_page,
                "automatic_page": automatic_page,
            }
            if fold == GUESSER_TRAIN_FOLD or fold == BUZZER_TRAIN_FOLD:
                train_unmatched_questions.append(q)
            else:
                test_unmatched_questions.append(q)
        elif (annotated_page is not None) and (automatic_page is None):
            q["page"] = annotated_page
            match_report[qanta_id] = {
                "result": "annotated",
                "annotated_error": annotated_error,
                "automatic_error": automatic_error,
                "annotated_page": annotated_page,
                "automatic_page": automatic_page,
            }
        elif (annotated_page is None) and (automatic_page is not None):
            q["page"] = automatic_page
            match_report[qanta_id] = {
                "result": "automatic",
                "annotated_error": annotated_error,
                "automatic_error": automatic_error,
                "annotated_page": annotated_page,
                "automatic_page": automatic_page,
            }
        else:
            if annotated_page == automatic_page:
                q["page"] = automatic_page
                match_report[qanta_id] = {
                    "result": "annotated+automatic",
                    "annotated_error": annotated_error,
                    "automatic_error": automatic_error,
                    "annotated_page": annotated_page,
                    "automatic_page": automatic_page,
                }
            else:
                q["page"] = annotated_page
                match_report[qanta_id] = {
                    "result": "disagree",
                    "annotated_error": annotated_error,
                    "automatic_error": automatic_error,
                    "annotated_page": annotated_page,
                    "automatic_page": automatic_page,
                }

    return {
        "train_unmatched": train_unmatched_questions,
        "test_unmatched": test_unmatched_questions,
        "match_report": match_report,
    }


def read_wiki_redirects(
    wiki_titles, redirect_csv_path=ALL_WIKI_REDIRECTS
) -> Dict[str, str]:
    with open(redirect_csv_path) as f:
        redirect_lookup = {}
        n = 0
        for source, target in csv.reader(f, escapechar="\\"):
            if target in wiki_titles:
                redirect_lookup[source] = target
            else:
                n += 1

        log.info(f"{n} titles in redirect not found in wiki titles")

        return redirect_lookup


def read_wiki_titles(title_path=WIKI_TITLES_PICKLE) -> Set[str]:
    with open(title_path, "rb") as f:
        return pickle.load(f)
