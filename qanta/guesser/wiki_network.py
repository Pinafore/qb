import re
import random
from collections import defaultdict
import networkx as nx
import pandas as pd
from nltk.corpus import stopwords
from qanta.util.environment import QB_QUESTION_DB
from qanta.datasets.quiz_bowl import QuestionDatabase


class WikiNetworkGuesser:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        g = nx.DiGraph()
        vertex_set = set()
        page_map = dict()
        with open('/ssd-c/qanta/titles-sorted.txt') as f:
            for i, line in enumerate(f, 1):
                page = line.strip().lower()
                if re.match("^[a-zA-Z_']+$", page) and page not in self.stop_words:
                    g.add_node(i, page=page)
                    vertex_set.add(i)
                    page_map[page] = i

        with open('/ssd-c/qanta/links-simple-sorted.txt') as f:
            for line in f:
                source, edges = line.split(':')
                source = int(source)
                if source in vertex_set:
                    edges = [int(vid) for vid in edges.split()]
                    for e in edges:
                        if e in vertex_set:
                            g.add_edge(source, e)
        self.g = g
        self.vertex_set = vertex_set
        self.page_map = page_map

    def tokenize(self, text):
        raw_words = text.lower().replace(',', '').replace('.', '').split()
        return [w for w in raw_words if w not in self.stop_words]

    def generate_guesses(self, text, answer, qnum):
        words = self.tokenize(text)
        seed_vertexes = {self.page_map[w] for w in words if w in self.page_map}
        candidate_vertexes = set()

        for seed_v in seed_vertexes:
            for v in self.g.neighbors(seed_v):
                if v not in seed_vertexes:
                    candidate_vertexes.add(v)

        seed_distances = defaultdict(int)
        for u in candidate_vertexes:
            for v in seed_vertexes:
                try:
                    seed_distances[u] += 1 / nx.shortest_path_length(self.g, source=u, target=v)
                except nx.NetworkXNoPath:
                    seed_distances[u] += 0

        columns = {'vid': [], 'distance': [], 'page': [], 'answer': [], 'qnum': []}

        for vid in candidate_vertexes:
            columns['vid'].append(vid)
            columns['page'].append(self.g.node[vid]['page'])
            columns['distance'].append(seed_distances[vid])
            columns['answer'].append(answer)
            columns['qnum'].append(qnum)

        return pd.DataFrame(columns), seed_vertexes


def evaluate():
    wiki = WikiNetworkGuesser()
    db = QuestionDatabase(QB_QUESTION_DB)
    questions = [q for q in db.all_questions().values() if q.fold == 'train']
    random.shuffle(questions)

    subset = questions[0:10]
    df = None
    for q in subset:
        tmp_df = wiki.generate_guesses(q.flatten_text(), q.page.lower().replace(' ', '_'), q.qnum)
        if df is None:
            df = tmp_df
        else:
            df = pd.concat([df, tmp_df])
    return df
