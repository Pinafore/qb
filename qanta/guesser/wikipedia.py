import re
import random
import os
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

    def tokenize(self, text):
        return text.lower().replace(',', '').replace('.', '').split()

    def build_subgraph(self, words):
        v_indexes = set()
        seed_vertexes = set()

        for w in words:
            if w in self.page_map:
                v = self.page_map[w]
                v_indexes.add(v)
                seed_vertexes.add(v)
                v_indexes |= set(self.g.neighbors(v))

        sub_graph = self.g.subgraph(v_indexes)
        size = 0
        max_size_subgraph = None
        for comp in nx.connected_component_subgraphs(sub_graph):
            if max_size_subgraph is None or len(comp) > size:
                max_size_subgraph = comp
                size = len(comp)

        seed_vertexes = {v for v in seed_vertexes if v in sub_graph.node}

        return max_size_subgraph, seed_vertexes

    def node2vec_input(self, answer: str, sub_graph: nx.Graph, output_directory):
        if answer not in self.page_map:
            print('No wiki entry for:', answer)
            return
        answer_id = self.page_map[answer]
        g_id = random.randint(0, 1000000)
        n2v_output = os.path.join(output_directory, '{}_edges.txt'.format(g_id))
        with open(n2v_output, 'w') as f:
            for u, v in sub_graph.edges_iter():
                f.write('{} {}\n'.format(u, v))

        meta_output = os.path.join(output_directory, '{}_meta.txt'.format(g_id))
        with open(meta_output, 'w') as f:
            for n in sub_graph.node:
                page = sub_graph.node[n]['page']
                if n == answer_id:
                    f.write('{} {} {}\n'.format(1, n, page))
                else:
                    f.write('{} {} {}\n'.format(0, n, page))

    def generate_guesses(self, text, answer, qnum):
        words = self.tokenize(text)
        sub_graph, seed_vertexes = self.build_subgraph(words)

        columns = {'vid': [], 'degree': [], 'page': [], 'answer': [], 'qnum': []}
        degree_dist = nx.degree(sub_graph)

        for vid in sub_graph.nodes():
            columns['vid'].append(vid)
            columns['page'].append(self.g.node[vid]['page'])
            columns['degree'].append(degree_dist[vid])
            columns['answer'].append(answer)
            columns['qnum'].append(qnum)

        return pd.DataFrame(columns), sub_graph, seed_vertexes


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
