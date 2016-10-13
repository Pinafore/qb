import re
import random
import networkx as nx
import pandas as pd
from qanta.util.environment import QB_QUESTION_DB
from qanta.util.qdb import QuestionDatabase


class WikiNetworkGuesser:
    def __init__(self):
        g = nx.Graph()
        vertex_set = set()
        page_map = dict()
        with open('/ssd-c/qanta/titles-sorted.txt') as f:
            for i, line in enumerate(f, 1):
                page = line.strip().lower()
                if re.match("^[a-zA-Z\_']+$", page):
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

    def generate_guesses(self, text, answer, qnum):
        words = text.lower().replace(',', '').replace('.', '').split()
        v_indexes = set()
        for w in words:
            if w in self.page_map:
                v = self.page_map[w]
                v_indexes.add(v)
                v_indexes = v_indexes | set(self.g.neighbors(v))

        sub_graph = self.g.subgraph(v_indexes)
        size = 0
        max_g = None
        for comp in nx.connected_component_subgraphs(sub_graph):
            if max_g is None or len(comp) > size:
                max_g = comp
                size = len(comp)

        columns = {'vid': [], 'degree': [], 'page': [], 'answer': [], 'qnum': []}
        degree_dist = nx.degree(max_g)

        for vid in max_g.nodes():
            columns['vid'].append(vid)
            columns['page'].append(self.g.node[vid]['page'])
            columns['degree'].append(degree_dist[vid])
            columns['answer'].append(answer)
            columns['qnum'].append(qnum)

        return pd.DataFrame(columns)


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
