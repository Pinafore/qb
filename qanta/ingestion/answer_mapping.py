from typing import Tuple, Set, Dict, List, Callable, Iterable, Optional
import shutil
import json
import re
import pickle
from os import path
from collections import defaultdict
from unidecode import unidecode
import sqlite3

from qanta import qlogging
from qanta.util.constants import WIKI_TITLES_PICKLE
from qanta.util.io import safe_open
from qanta.datasets.quiz_bowl import QuestionDatabase


log = qlogging.get(__name__)

ExpansionRule = Callable[[str], Iterable[str]]
MatchRule = Callable[[str], Optional[str]]


def create_answer_map(
        expansion_rules: List[Tuple[str, ExpansionRule]],
        match_rules: List[Tuple[str, MatchRule]],
        lower_titles: Dict[str, str], unicode_titles: Dict[str, str],
        unmapped_answers: Set[str]):
    answer_map = {}

    # Clone the set to prevent accidental mutation of the original
    unmapped_answers = set(unmapped_answers)

    original_num = len(unmapped_answers)
    log.info(f'{original_num} Unmapped Answers Exist\nStarting Answer Mapping\n')

    expansion_answer_map = defaultdict(set)
    for ans in unmapped_answers:
        expansion_answer_map[ans].add(ans)

    for name, rule_func in expansion_rules:
        log.info(f'Applying expansion rule: {name}')
        for raw_ans in unmapped_answers:
            for exp_ans in rule_func(raw_ans):
                expansion_answer_map[raw_ans].add(exp_ans.strip())

    for name, rule_func in match_rules:
        curr_num = len(unmapped_answers)
        log.info(f'Applying rule: {name}')

        for original_ans, ans_expansions in expansion_answer_map.items():
            for raw_ans in ans_expansions:
                rule_ans = re.sub(r'\s+', ' ', rule_func(raw_ans).strip())
                if rule_ans is None:
                    continue
                else:
                    mod_ans = rule_ans.lower()
                    und_mod_ans = mod_ans.replace(' ', '_')

                    # continue statements: We only need at least one expansion to match.
                    # Once we find it we can skip looking at the others
                    if mod_ans in lower_titles:
                        answer_map[original_ans] = lower_titles[mod_ans]
                        continue
                    elif und_mod_ans in lower_titles:
                        answer_map[original_ans] = lower_titles[und_mod_ans]
                        continue
                    else:
                        pass

                    if mod_ans in unicode_titles:
                        answer_map[original_ans] = unicode_titles[mod_ans]
                        continue
                    elif und_mod_ans in unicode_titles:
                        answer_map[original_ans] = unicode_titles[und_mod_ans]
                        continue
                    else:
                        pass

        unmapped_answers -= set(answer_map.keys())
        removed_num = curr_num - len(unmapped_answers)
        log.info(f'{removed_num} Answers Mapped')
        log.info(f'{len(unmapped_answers)} remain\n')

    end_num = len(unmapped_answers)
    log.info(f'\nAnswer Mapping Complete\n{end_num} Unmapped Remain, {len(answer_map)} Mappings Found')

    return answer_map, unmapped_answers


# Expansion rule functions
def or_rule(ans):
    splits = ans.split('or')
    if len(splits) > 1:
        formatted_splits = [s.strip() for s in splits]
        return formatted_splits
    else:
        return ()


def prompt_rule(ans):
    ans = ans.lower()
    if 'accept' in ans or 'prompt' in ans or 'pronounce' in ans:
        m = re.match(r'(.+)\(.*(?:accept|prompt|pronounce).*\)', ans)
        if m is not None:
            return (m.group(1).strip(),)

        m = re.match(r'(.+)\[.*(?:accept|prompt|pronounce).*\]', ans)
        if m is not None:
            return (m.group(1).strip(),)

        return ()
    elif 'or' in ans:
        m = re.match(r'(.+)\(.*(?:or).*\)', ans)
        if m is not None:
            return (m.group(1).strip(),)

        m = re.match(r'(.+)\[.*(?:or).*\]', ans)
        if m is not None:
            return (m.group(1).strip(),)

        return ()
    else:
        return ()


def the_rule(ans):
    l_ans = ans.lower()
    if 'the' in l_ans:
        return (l_ans.replace('the', ''),)
    else:
        return ('the ' + l_ans.lower(),)


def plural_rule(ans):
    if ans.endswith('s'):
        return ans[:-1]
    else:
        return ans


def apostraphe_rule(ans):
    if "’" in ans:
        return (ans.replace('’', "'"), ans.replace('’', ''))
    else:
        return ()


def answer_rule(ans):
    l_ans = ans.lower()
    if 'answers:' in l_ans:
        return (l_ans.replace('answers:', ''),)
    elif 'answer:' in l_ans:
        return (l_ans.replace('answer:', ''),)
    else:
        return ()


def optional_text_rule(ans):
    candidate = re.sub(r'\(.+?\)', '', ans)
    if candidate != ans:
        return (candidate,)
    else:
        return ()


def parens_rule(ans):
    if "(" in ans and ")" in ans:
        return (ans.replace('(', '').replace(')', ''),)
    else:
        return ()


def sir_rule(ans):
    if 'sir' in ans.lower():
        return (ans.lower().replace('sir', ''),)
    else:
        return ()


def unicode_rule(ans):
    unicode_ans = unidecode(ans)
    if ans != unicode_ans:
        return (unicode_ans,)
    else:
        return ()


# Match Rule Functions
def remove_braces(text):
    return re.sub(r'[{}]', '', text)


def remove_quotes(text):
    return re.sub(r'["“”]', '', text)


def remove_parens(text):
    return re.sub(r'[\(\)]', '', text)


def compose(*funcs):
    def composed_function(x):
        for f in funcs:
            x = f(x)
        return x

    return composed_function


def create_expansion_rules():
    # Apply this rules to generate multiple possible answers from one distinct answer
    expansion_rules = [
        ('or', or_rule),
        ('the', the_rule),
        ('prompt', prompt_rule),
        ('apostraphe', apostraphe_rule),
        ('parens', parens_rule),
        ('unicode', unicode_rule),
        ('sir', sir_rule),
        ('answer', answer_rule),
        ('optional-text', optional_text_rule)
    ]
    return expansion_rules


def create_match_rules():
    # Take an answer, format it, then check if there is an exact match
    match_rules = [
        ('exact match', lambda x: x),
        ('braces', remove_braces),
        ('quotes', remove_quotes),
        ('parens', remove_parens),
        ('braces+quotes', compose(remove_braces, remove_quotes)),
        ('braces+plural', compose(remove_braces, plural_rule)),
        ('quotes+braces+plural', compose(remove_braces, remove_quotes, plural_rule))
    ]
    return match_rules


def write_answer_map(output_dir):
    expansion_rules = create_expansion_rules()
    match_rules = create_match_rules()

    log.info('Loading questions')
    db = QuestionDatabase()
    question_lookup = db.all_questions(unfiltered=True)
    questions = list(question_lookup.values())
    unmapped_questions = [q for q in questions if q.page == '']
    raw_unmapped_answers = {q.answer for q in unmapped_questions}
    unmapped_lookup = defaultdict(list)
    for q in unmapped_questions:
        unmapped_lookup[q.answer].append(q)

    log.info('Loading wikipedia titles')
    with open(WIKI_TITLES_PICKLE, 'rb') as f:
        titles = pickle.load(f)
        lower_title_map = {t.lower(): t for t in titles}
        unicode_title_map = {unidecode(t.lower()): t for t in titles}

    log.info('Starting Answer Mapping Process')
    answer_map, unbound = create_answer_map(
        expansion_rules, match_rules,
        lower_title_map, unicode_title_map,
        raw_unmapped_answers
    )

    answer_map_path = path.join(output_dir, 'answer_map.json')
    log.info(f'Writing answer map to: {answer_map_path}')
    with safe_open(answer_map_path, 'w') as f:
        json.dump({'answer_map': answer_map}, f)

    unbound_path = path.join(output_dir, 'unbound.json')
    log.info(f'Writing unbound answers to: {unbound_path}')
    with safe_open(unbound_path, 'w') as f:
        json.dump({'unbound': list(sorted(unbound))}, f)


def merge_answer_mapping(source_db_path, answer_map, output_db_path, page_assignments_path):
    shutil.copyfile(source_db_path, output_db_path)
    conn = sqlite3.connect(output_db_path)
    c = conn.cursor()
    questions = list(c.execute("select id, answer from questions where page=''"))
    page_assignments = []
    for qnum, answer in questions:
        if answer in answer_map:
            page = answer_map[answer]
            page_assignments.append((page, qnum))

    update_sql = """
        UPDATE questions
        SET page = ?
        WHERE
          id = ?;
    """

    c.executemany(update_sql, page_assignments)
    conn.commit()
    conn.close()

    with safe_open(page_assignments_path, 'w') as f:
        json.dump({'page_assignments': page_assignments}, f)

