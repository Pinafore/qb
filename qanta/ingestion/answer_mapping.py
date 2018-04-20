from typing import Tuple, Set, Dict, List, Callable, Iterable, Optional
import csv
import json
import re
import pickle
from collections import defaultdict
from unidecode import unidecode

from qanta import qlogging
from qanta.util.constants import WIKI_TITLES_PICKLE, ALL_WIKI_REDIRECTS
from qanta.util.io import safe_open


log = qlogging.get(__name__)

ExpansionRule = Callable[[str], Iterable[str]]
MatchRule = Callable[[str], Optional[str]]


def mapping_rules_to_answer_map(
        expansion_rules: List[Tuple[str, ExpansionRule]],
        match_rules: List[Tuple[str, MatchRule]],
        lower_titles: Dict[str, str], unicode_titles: Dict[str, str],
        lower_wiki_redirects: Dict[str, str], unicode_wiki_redirects: Dict[str, str],
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

                    if mod_ans in lower_wiki_redirects:
                        answer_map[original_ans] = lower_wiki_redirects[mod_ans]
                        continue
                    elif und_mod_ans in lower_wiki_redirects:
                        answer_map[original_ans] = lower_wiki_redirects[und_mod_ans]
                        continue
                    else:
                        pass

                    if mod_ans in unicode_wiki_redirects:
                        answer_map[original_ans] = unicode_wiki_redirects[mod_ans]
                        continue
                    elif und_mod_ans in unicode_wiki_redirects:
                        answer_map[original_ans] = unicode_wiki_redirects[und_mod_ans]
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


def create_answer_map(unmapped_qanta_questions):
    expansion_rules = create_expansion_rules()
    match_rules = create_match_rules()

    log.info('Loading questions')
    raw_unmapped_answers = {q['answer'] for q in unmapped_qanta_questions}
    unmapped_lookup = defaultdict(list)
    for q in unmapped_qanta_questions:
        unmapped_lookup[q['answer']].append(q)

    log.info('Loading wikipedia titles')
    with open(WIKI_TITLES_PICKLE, 'rb') as f:
        titles = pickle.load(f)
        lower_title_map = {t.lower(): t for t in titles}
        unicode_title_map = {unidecode(t.lower()): t for t in titles}

    wiki_redirect_map = read_wiki_redirects(titles)
    lower_wiki_redirect_map = {text.lower(): page for text, page in wiki_redirect_map.items()}
    unicode_wiki_redirect_map = {unidecode(text.lower()): page for text, page in wiki_redirect_map.items()}

    log.info('Starting Answer Mapping Process')
    answer_map, unbound_answers = mapping_rules_to_answer_map(
        expansion_rules, match_rules,
        lower_title_map, unicode_title_map,
        lower_wiki_redirect_map, unicode_wiki_redirect_map,
        raw_unmapped_answers
    )
    return answer_map, unbound_answers


def write_answer_map(answer_map, unbound_answers, answer_map_path, unbound_answer_path):
    with safe_open(answer_map_path, 'w') as f:
        json.dump({'answer_map': answer_map}, f)

    with safe_open(unbound_answer_path, 'w') as f:
        json.dump({'unbound_answers': list(sorted(unbound_answers))}, f)


def unmapped_to_mapped_questions(unmapped_qanta_questions, answer_map):
    for q in unmapped_qanta_questions:
        if q['answer'] in answer_map:
            q['page'] = answer_map[q['answer']]


def read_wiki_redirects(wiki_titles, redirect_csv_path=ALL_WIKI_REDIRECTS) -> Dict[str, str]:
    with open(redirect_csv_path) as f:
        redirect_lookup = {}
        n = 0
        for source, target in csv.reader(f, escapechar='\\'):
            if target in wiki_titles:
                redirect_lookup[source] = target
            else:
                n += 1

        log.info(f'{n} titles in redirect not found in wiki titles')

        return redirect_lookup
