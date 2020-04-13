import json
from functional import pseq


def fetch_parse(sentence):
    if sentence is None:
        return None
    properties = json.dumps({
        'annotators': 'tokenize,ssplit,parse',
        'date': "2018-09-12T16:34:00"
    })
    url = f'http://quenya.umiacs.umd.edu:9000/?properties={properties}&pipelineLanguage=en'
    try:
        response = requests.post(url, data=sentence).json()
    except:
        return None

    annotated = response['sentences']
    if len(annotated) > 0:
        return annotated[0]['parse']
    else:
        return None

def collect_parses(question_sentences):
    return pseq(question_sentences).map(fetch_parse).list()

    
def process_sq_question(raw_question):
    question = raw_question.replace('.', '').replace('?', '')
    return question

def process_tqa_question(raw_question):
    question = raw_question.replace('.', '').replace('?', '')
    if question[0] == '"' and question[-1] == '"':
        question = re.sub(r'""[^("")]+""', 'QUOTETOKEN', question[1:-1]).replace('"', '')
    question = re.sub(r'\s+', ' ', question).strip()
    return question

def process_squad_question(raw_question):
    question = raw_question.replace('.', '').replace('?', '')
    question = re.sub(r'"[^"]+"', 'QUOTETOKEN', question)
    question = re.sub(r'\s+', ' ', question).strip()
    return question

def process_qb_question(raw_question):
    if len(raw_question) < 2:
        return None
    
    question = raw_question.replace('.', '').replace('/', '')
    
    if question.startswith('" '):
        question = question[2:]
        
    question = re.sub(r'"[^"]+"', 'QUOTETOKEN', question)
    if question.endswith(' "'):
        question = question[:-1]
    question = question.replace('"', '')
    question = re.sub(r'\s+', ' ', question).strip()
    return question

def process_jeopardy_question(raw_question):
    question = raw_question.replace(':', ' ').replace('.', '')
    question = question.replace('&', 'and')
    if question[0] == "'" and question[-1] == "'":
        question = question[1:-1]
    if re.match(r'[0-9,](?:\sx\s[0-9,]+)+', question):
        return None
    
    if '<a' in question or '<i>' in question or '<br' in question:
        question = BeautifulSoup(question, 'lxml').get_text()

    question = re.sub(r'\([^\(\)]+\)', ' ', question)
    question = re.sub(r'"[^"]+"', 'QUOTETOKEN', question)
    question = re.sub(r'\s+', ' ', question).strip()
    return question


# On the machine the following cells were run, corenlp was running in server mode.
# Running all of these runs the machine out of ram, so for it to work you need to
# run each dataset parse, save it, then possibly restart the session and continue
# with other datasets. Its hacky, but it works.... so there it is
qb_parses = collect_parses([process_qb_question(q) for q in qb_sentences])
with open('data/external/qb_parses.json', 'w') as f:
    json.dump(qb_parses, f)

sq_parses = collect_parses([process_sq_question(q) for q in sq_questions])
with open('data/external/simplequestions_parses.json', 'w') as f:
    json.dump(sq_parses, f)

squad_parses = collect_parses([process_squad_question(q) for q in squad_questions])

with open('data/external/squad_parses.json', 'w') as f:
    json.dump(squad_parses, f)

tqa_parses = collect_parses([process_tqa_question(q) for q in tqa_questions])


with open('data/external/tqa_parses.json', 'w') as f:
    json.dump(tqa_parses, f)

jeopardy_parses = collect_parses([process_jeopardy_question(q) for q in j_questions])

with open('data/external/jeopardy_parses.json', 'w') as f:
    json.dump(jeopardy_parses, f)

with open('data/external/qb_parses.json') as f:
    qb_parses = json.load(f)
with open('data/external/simplequestions_parses.json') as f:
    sq_parses = json.load(f)
with open('data/external/squad_parses.json') as f:
    squad_parses = json.load(f)
with open('data/external/tqa_parses.json') as f:
    tqa_parses = json.load(f)
with open('data/external/jeopardy_parses.json') as f:
    jeopardy_parses = json.load(f)

from pyparsing import Forward, Literal, Group, OneOrMore, Regex, ParseException
import sys
from result import Ok, Err
import tqdm

class Term:
    __slots__ = ('term',)
    def __init__(self, term):
        self.term = term
    
    def __repr__(self):
        return f'Term({self.term})'

lpar = Literal('(').suppress()
rpar = Literal(')').suppress()
part = lpar + Regex('[^\s\(]+')
leaf = Regex('[^\s\)]+').setParseAction( lambda s, l, t: [Term(t[0])])
node = Forward()
root = Group(part + leaf + rpar)
node << OneOrMore(root | Group(part + node + rpar))
node.enablePackrat()


def try_parse(corenlp_parse):
    if corenlp_parse is None:
        return None
    try:
        return node.parseString(corenlp_parse).asList()
    except ParseException:
        return None

qb_parse_results = pseq(qb_parses).map(try_parse).list()
sq_parse_results = pseq(sq_parses).map(try_parse).list()
squad_parse_results = pseq(squad_parses).map(try_parse).list()
tqa_parse_results = pseq(tqa_parses).map(try_parse).list()
jeopardy_parse_results = pseq(jeopardy_parses).map(try_parse).list()
def compute_pcfg(parse_results):
    non_terminals = Counter()
    transitions = defaultdict(Counter)
    for tree in parse_results:
        if tree is not None:
            root = tree[0]
            update_transitions(root, non_terminals, transitions)
    return non_terminals, transitions

def update_transitions(tree, non_terminals, transitions):
    left = tree[0]
    right = tree[1:]
    if not isinstance(left, str):
        raise ValueError(f'Invalid left expression: {left}')
    non_terminals[left] += 1
    for sub_tree in right:
        if isinstance(sub_tree, list):
            sub_tree_types = {type(t) for t in sub_tree}
            if len(sub_tree_types) > 2:
                raise ValueError(f'Bad types: {sub_tree_types}')
            elif Term in sub_tree_types:
                continue
            elif (list in sub_tree_types) and (str in sub_tree_types):
                transition_terms = [t[0] for t in sub_tree]
                transitions[left][' '.join(transition_terms)] += 1
                update_transitions(sub_tree, non_terminals, transitions)
            else:
                raise ValueError(f'Unknown type: {sub_tree_types}')
        else:
            raise ValueError(f'Invalid type for: {sub_tree}')

def compute_parse_entropy(parse_results):
    non_terminal_dist, transition_dist = compute_pcfg(parse_results)
    transition_probs = {}
    total = sum(non_terminal_dist.values())
    for left, lookup in transition_dist.items():
        for right, count in lookup.items():
            transition_probs[(left, right)] = count / non_terminal_dist[left]
    normalized_probs = {}
    for (left, right), p in transition_probs.items():
        normalized_probs[(left, right)] = p * (non_terminal_dist[left] / total)
    entropy = -sum(p * np.log(p) for p in normalized_probs.values())
    return entropy

compute_parse_entropy(qb_parse_results)
compute_parse_entropy(sq_parse_results)
compute_parse_entropy(squad_parse_results)
compute_parse_entropy(tqa_parse_results)
compute_parse_entropy(jeopardy_parse_results)
class Box:
    __slots__ = ('value',)
    def __init__(self, value):
        self.value = value

def _linearize_parse(parse, depth=sys.maxsize):
    if depth == 0:
        return None
    else:
        if type(parse) == str:
            return f'({parse})'
        elif type(parse) == Term:
            return None
        else:
            phrase = parse[0]
            arguments = [_linearize_parse(a, depth=depth - 1) for a in parse[1:]]
            arguments = [a for a in arguments if a is not None]
            if len(arguments) == 0:
                arg_str = ''
            else:
                arg_str = ' ' + ' '.join(arguments)
            return f'({phrase}{arg_str})'

class LinearizeParse:
    def __init__(self, depth=sys.maxsize):
        self.depth = depth
    
    def __call__(self, parse_result):
        # Unwrap 0th level, select argument to ROOT
        if parse_result is None:
            return None
        else:
            base = parse_result[0][1]
            parse_str = _linearize_parse(base, depth=self.depth)
            if parse_str is None:
                return '()'
            else:
                return parse_str

def parses_to_linears(linearize_parse, parses, questions):
    proc_parses = [linearize_parse(p) for p in parses]
    lengths = [len(q.split()) for q in questions]
    if len(proc_parses) != len(lengths):
        raise ValueError('unequal lengths')
    final_parses = []
    final_lengths = []
    for p, l in zip(proc_parses, lengths):
        if p is not None:
            final_parses.append(p)
            final_lengths.append(l)
    return final_parses, final_lengths

deepest_parse_depth = 21
qb_length_results = {}
sq_length_results = {}
squad_length_results = {}
tqa_length_results = {}
jeopardy_length_results = {}
rows = []
for depth in tqdm.tqdm_notebook(range(1, deepest_parse_depth)):
    linearize_parse = LinearizeParse(depth=depth)
    qb_linear, qb_lengths = parses_to_linears(linearize_parse, qb_parse_results, qb_sentences)
    sq_linear, sq_lengths = parses_to_linears(linearize_parse, sq_parse_results, sq_questions)
    squad_linear, squad_lengths = parses_to_linears(linearize_parse, squad_parse_results, squad_questions)
    tqa_linear, tqa_lengths = parses_to_linears(linearize_parse, tqa_parse_results, tqa_questions)
    jeopardy_linear, jeopardy_lengths = parses_to_linears(linearize_parse, jeopardy_parse_results, j_questions)
    
    qb_length_results[depth] = qb_linear, qb_lengths
    sq_length_results[depth] = sq_linear, sq_lengths
    squad_length_results[depth] = squad_linear, squad_lengths
    tqa_length_results[depth] = tqa_linear, tqa_lengths
    jeopardy_length_results[depth] = jeopardy_linear, jeopardy_lengths
    
    qb_set = set(qb_linear)
    sq_set = set(sq_linear)
    squad_set = set(squad_linear)
    tqa_set = set(tqa_linear)
    jeopardy_set = set(jeopardy_linear)
    
    rows.append({
        'depth': depth,
        'unique_parses': len(qb_set),
        'parses': len(qb_linear),
        'dataset': 'Quizbowl',
        'overlap': 1
    })
    rows.append({
        'depth': depth,
        'unique_parses': len(sq_set),
        'parses': len(sq_linear),
        'dataset': 'SimpleQuestions',
        'overlap': len(qb_set & sq_set) / len(sq_set)
    })
    rows.append({
        'depth': depth,
        'unique_parses': len(squad_set),
        'parses': len(squad_linear),
        'dataset': 'SQuAD',
        'overlap': len(qb_set & squad_set) / len(squad_set)
    })
    rows.append({
        'depth': depth,
        'unique_parses': len(tqa_set),
        'parses': len(tqa_linear),
        'dataset': 'TriviaQA',
        'overlap': len(qb_set & tqa_set) / len(tqa_set)
    })
    rows.append({
        'depth': depth,
        'unique_parses': len(jeopardy_set),
        'parses': len(jeopardy_linear),
        'dataset': 'Jeopardy!',
        'overlap': len(qb_set & jeopardy_set) / len(jeopardy_set)
    })

with open('data/external/syntactic_diversity_table.json', 'w') as f:
    json.dump(rows, f)