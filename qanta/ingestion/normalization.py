import json
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
    'Mavis Gallant Memorial': 'Mavis Gallant Memorial Tournament (Literature)'
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
        with open(path) as f:
            quizdb_subcategory_list = json.load(f)
            quizdb_subcategories = {
                r['id']: r['name'] for r in quizdb_subcategory_list
            }
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
                quizdb_questions.append({
                    'text': q['text'],
                    'answer': q['answer'],
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
                protobowl_questions.append({
                    'text': q['question'],
                    'answer': q['answer'],
                    'category': q['category'],
                    'subcategory': q['subcategory'],
                    'tournament': q['tournament'],
                    'difficulty': q['difficulty'],
                    'year': q['year'],
                    'proto_id': q['num'],
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
    for q in itertools.chain(protobowl_questions, quizdb_questions):
        ty = (q['tournament'], q['year'])
        if ty in selected_tournaments:
            is_proto = selected_tournaments[ty][0].startswith('proto')
            is_qdb = selected_tournaments[ty][0].startswith('qdb')
            if is_proto and q['dataset'] == 'protobowl':
                questions.append(q)
            elif is_qdb and q['dataset'] == 'quizdb.org':
                questions.append(q)

    return questions
