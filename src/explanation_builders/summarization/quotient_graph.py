import itertools

import networkx as nx

# from networkx.drawing.nx_agraph import to_agraph


class QuotientGraph:
    def build_quotient_graph(self, graph, partition, condition=any):
        triples = set(graph.edges(data="label"))
        has_edge = lambda u, v, label: (u, v, label) in triples
        quotient = nx.MultiDiGraph()

        for part in partition:
            part = sorted(part)
            node_labels = [self.dataset.id_to_entity[node] for node in part]
            label = "\n".join(node_labels)
            quotient.add_node(frozenset(part), **{"label": label})

        for U, V in itertools.product(quotient.nodes, repeat=2):
            labels = [rel for head, tail, rel in triples if (head in U and tail in V)]

            for label in set(labels):
                if condition(any(has_edge(u, v, label) for v in V) for u in U):
                    quotient.add_edge(
                        U, V, label=label, id=self.dataset.relation_to_id[label]
                    )
        return quotient

    # def plot(self, graph, filename="graph", format="svg", quotient=False):
    #     agraph = to_agraph(graph)
    #     agraph.graph_attr["rankdir"] = "LR"
    #     agraph.graph_attr["pad"] = 0.01

    #     for node in agraph.nodes():
    #         node.attr["shape"] = "rectangle"
    #         node.attr["style"] = "rounded"
    #     for edge in agraph.edges():
    #         edge.attr["arrowsize"] = 0.3
    #         edge.attr["color"] = "red"

    #     agraph.draw(f"pictures/{filename}.{format}", prog="dot", format=format)

    def set_quotient_triple_to_triples(self, q_triples, filter):
        self.quotient_triple_to_triples = {}

        for s_part, p, o_part in q_triples:
            triples = [(s, p, o) for s in s_part for o in o_part]
            triples = [triple for triple in triples if triple in filter]

            if len(triples) > 0:
                self.quotient_triple_to_triples[(s_part, p, o_part)] = triples

    def map_rule(self, rule):
        triples = []
        for q_triple in rule:
            triples += self.quotient_triple_to_triples[q_triple]
        return triples
