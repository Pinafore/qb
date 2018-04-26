import json
import random
import re
import itertools
from collections import Counter


def try_parse_int(text):
    try:
        return int(text)
    except:
        return None


CANONICAL_TOURNAMENT_MAP = {
    'EFT': 'Early Fall Tournament (EFT)',
    'FKT': 'Fall Kickoff Tournament (FKT)',
    'Fall Kickoff Tournament': 'Fall Kickoff Tournament (FKT)',
    'LIST': 'Ladue Invitational Sprint Tournament (LIST)',
    'LIST (Ladue Invitational Spring Tournament)': 'Ladue Invitational Sprint Tournament (LIST)',
    'LIST (Ladue Invitational Spring Tournament) VI': 'Ladue Invitational Sprint Tournament (LIST)',
    'LIST III': 'Ladue Invitational Sprint Tournament (LIST)',
    'LIST IV': 'Ladue Invitational Sprint Tournament (LIST)',
    'Ladue Invitational Spring Tournament': 'Ladue Invitational Sprint Tournament (LIST)',
    'Maggie Walker GSAC XIX': 'Maggie Walker GSAC',
    'Maggie Walker GSAC XV': 'Maggie Walker GSAC',
    'Maggie Walker GSAC XVI': 'Maggie Walker GSAC',
    'Maggie Walker GSAC XVIII': 'Maggie Walker GSAC',
    'Prison Bowl VIII': 'Prison Bowl',
    'Prison Bowl X': 'Prison Bowl',
    'Tyrone Slothrop Lit': 'Tyrone Slothrop Literature Singles',
    'Terrapin': 'Terrapin Invitational Tournament',
    'Terrapin Invitational': 'Terrapin Invitational Tournament',
    'Mavis Gallant Memorial': 'Mavis Gallant Memorial Tournament (Literature)',
    'Geography Monstrosity 4': 'Geography Monstrosity',
    'Geography Monstrosity 2': 'Geography Monstrosity'
}


def parse_tournament_name(tournament_name):
    splits = tournament_name.split()
    maybe_year = try_parse_int(splits[0])
    if maybe_year is None:
        if tournament_name in CANONICAL_TOURNAMENT_MAP:
            return CANONICAL_TOURNAMENT_MAP[tournament_name], None
        return tournament_name, None
    else:
        name = ' '.join(splits[1:])
        if name in CANONICAL_TOURNAMENT_MAP:
            return CANONICAL_TOURNAMENT_MAP[name], maybe_year
        else:
            return name, maybe_year

TRASH_PREFIXES = [
    r'.*\(Note to moderator:.*\)',
    r'\.', r'\?', r'\|', r'\_', r'\)',
    r'[0-9]+[\.:]?', 'C:', r'\[[A-Z/]+\]', r'\([A-Z/]+\)', r'BONUS\.?',
    '10 pts:', '15 pts:', '10 points:', '15 points:', 'Round [0-9]+:',
    'BHSAT 2008 Packet #[0-9]+ Packet by Robert, Ian, Danila, and Linna',
    r'Two answers required\.', r"The name's the same\.", 'TWO ANSWERS REQUIRED\.',
    'Warning: two answers required\.',
    'NOTE:', 'WARNING:', 'MODERATOR NOTE:',
    r'Pencil and paper ready\.',
    r'\([A-Z]+\) Computational - pencil and paper ready\.',
    r'Description acceptable\.',
    r'Pyramidal Math \([0-9]+ Seconds\)',
    r'Physics \([0-9]+ Seconds\)',
    'Chemistry', 'Nonfiction', 'Vocabulary', 'US History', 'Music', 'Biology',
    'Art/Architecture', 'Art/Archictecture', 'World Literature',
    'Interdisciplinary', 'British Literature', 'Religion/Mythology',
    'Tiebreaker [0-9]+.', 'Pop Culture', 'US Literature', 'World History',
    r'Pencil and Paper Ready\.', 'United States History', 'United States Literature',
    'Geography/Earth Science/Astronomy', 'Geography/Astronomy/Earth Science',
    'Extra Tossups', 'Current Events',
    'Extra Toss-Up #[0-9]+', 'Toss-Up #[0-9]+'
]
TRASH_PREFIX_PATTERN = '^({})'.format('|'.join(TRASH_PREFIXES))


def normalize_text(text):
    text = text.replace('“','"').replace('”','"').replace('’', "'")
    return re.sub(TRASH_PREFIX_PATTERN, '', text).lstrip()


class QuizdbOrg:
    @staticmethod
    def parse_tournaments(path):
        with open(path) as f:
            quizdb_tournaments = {}
            for r in json.load(f):
                name, year = parse_tournament_name(r['name'])
                if year is not None and r['year'] != year:
                    raise ValueError('Years disagree, thats unexpected')
                quizdb_tournaments[r['id']] = {
                    'name': name,
                    'year': r['year'],
                    'difficulty': r['difficulty']
                }
            return quizdb_tournaments

    @staticmethod
    def parse_categories(path):
        with open(path) as f:
            quizdb_category_list = json.load(f)
            quizdb_categories = {
                r['id']: r['name'] for r in quizdb_category_list
            }
            return quizdb_categories

    @staticmethod
    def parse_subcategories(path):
        categories = [
            'Current Events', 'Fine Arts', 'Geography',
            'History', 'Literature', 'Mythology', 'Philosophy',
            'Religion', 'Science', 'Social Science', 'Trash'
        ]
        pattern = f"(?:{'|'.join(categories)}) (.*)"
        with open(path) as f:
            quizdb_subcategory_list = json.load(f)
            quizdb_subcategories = {}
            for r in quizdb_subcategory_list:
                m = re.match(pattern, r['name'])
                if m is None:
                    quizdb_subcategories[r['id']] = r['name']
                else:
                    quizdb_subcategories[r['id']] = m.group(1)

            return quizdb_subcategories

    @staticmethod
    def parse_tossups(qdb_tournaments, qdb_categories, qdb_subcategories, path):
        with open(path) as f:
            quizdb_questions = []
            for q in json.load(f):
                category_id = q['category_id']
                subcategory_id = q['subcategory_id']
                tournament_id = q['tournament_id']
                if tournament_id is None:
                    tournament = None
                    difficulty = None
                    year = -1
                else:
                    t = qdb_tournaments[tournament_id]
                    tournament = t['name']
                    difficulty = t['difficulty']
                    year = int(t['year'])
                if q['text'] == '[missing]':
                    continue
                quizdb_questions.append({
                    'text': normalize_text(q['text']),
                    'answer': q['answer'],
                    'page': None,
                    'category': qdb_categories[category_id] if category_id is not None else None,
                    'subcategory': qdb_subcategories[subcategory_id] if subcategory_id is not None else None,
                    'tournament': tournament,
                    'difficulty': difficulty,
                    'year': year,
                    'proto_id': None,
                    'qdb_id': q['id'],
                    'dataset': 'quizdb.org'
                })
            return quizdb_questions


class Protobowl:
    @staticmethod
    def parse_tossups(path):
        with open(path) as f:
            protobowl_raw = [json.loads(l) for l in f]
            protobowl_questions = []
            for q in protobowl_raw:
                if q['question'] == '[missing]':
                    continue
                protobowl_questions.append({
                    'text': normalize_text(q['question']),
                    'answer': q['answer'],
                    'page': None,
                    'category': q['category'],
                    'subcategory': q['subcategory'],
                    'tournament': q['tournament'],
                    'difficulty': q['difficulty'],
                    'year': q['year'],
                    'proto_id': q['_id']['$oid'],
                    'qdb_id': None,
                    'dataset': 'protobowl'
                })
            return protobowl_questions


def merge_datasets(protobowl_questions, quizdb_questions):
    """
    This function is responsible for merging protobowl and quizdb datasets. The primary steps
    in this process are:
    1) Compute a list of tournament/year
    2) Select which dataset to get questions from for a specific tournament and year
    3) Return the dataset in json serializable format

    :param protobowl_questions: Parsed protobowl questions
    :param quizdb_questions: Parsed quizdb questions
    :return:
    """
    proto_tournament_years = Counter()
    for r in protobowl_questions:
        if r['tournament'] is not None:
            proto_tournament_years[(r['tournament'], r['year'])] += 1

    qdb_tournament_years = Counter()
    for r in quizdb_questions:
        if r['tournament'] is not None:
            qdb_tournament_years[(r['tournament'], r['year'])] += 1

    selected_tournaments = {}
    possible_tournaments = set(qdb_tournament_years.keys()) | set(proto_tournament_years.keys())
    for ty in possible_tournaments:
        if ty in proto_tournament_years and ty in qdb_tournament_years:
            n_proto = proto_tournament_years[ty]
            n_qdb = qdb_tournament_years[ty]
            n_max = max(n_proto, n_qdb)
            n_min = min(n_proto, n_qdb)
            p_10 = .1 * n_max
            if n_proto > n_qdb:
                selected_tournaments[ty] = ('proto_choose', n_proto, n_qdb)
            elif (n_max - n_min) <= p_10:
                selected_tournaments[ty] = ('proto_close', n_proto, n_proto)
            else:
                selected_tournaments[ty] = ('qdb_choose', n_qdb, n_proto)
        elif ty in proto_tournament_years:
            selected_tournaments[ty] = ('proto_default', proto_tournament_years[ty], 0)
        elif ty in qdb_tournament_years:
            selected_tournaments[ty] = ('qdb_default', qdb_tournament_years[ty], 0)
        else:
            raise ValueError('This is impossible')

    questions = []
    for i, q in enumerate(itertools.chain(protobowl_questions, quizdb_questions)):
        ty = (q['tournament'], q['year'])
        if ty in selected_tournaments:
            is_proto = selected_tournaments[ty][0].startswith('proto')
            is_qdb = selected_tournaments[ty][0].startswith('qdb')
            if is_proto and q['dataset'] == 'protobowl':
                q['qanta_id'] = i
                questions.append(q)
            elif is_qdb and q['dataset'] == 'quizdb.org':
                q['qanta_id'] = i
                questions.append(q)

    return questions


TEST_TOURNAMENTS = {'ACF Regionals', 'PACE NSC', 'NASAT', 'ACF Nationals', 'ACF Fall'}
TEST_YEARS = {2017, 2018}
DEV_YEARS = {2016}


def assign_folds(qanta_questions, random_seed=0, guessbuzz_frac=.8):
    random.seed(random_seed)
    for q in qanta_questions:
        if q['tournament'] in TEST_TOURNAMENTS and q['year'] in TEST_YEARS:
            if random.random() < .5:
                q['fold'] = 'test'
            else:
                q['fold'] = 'dev'
        elif q['tournament'] in TEST_TOURNAMENTS and q['year'] in DEV_YEARS:
            if q['dataset'] == 'protobowl':
                q['fold'] = 'buzzdev'
            else:
                q['fold'] = 'guessdev'
        else:
            if random.random() < guessbuzz_frac:
                q['fold'] = 'guesstrain'
            else:
                q['fold'] = 'buzztrain'
