import networkx as nx

from bispy import compute_maximum_bisimulation

from .quotient_graph import QuotientGraph

is_tuple = lambda node: isinstance(node, tuple)
contain_tuples = lambda cl: any(is_tuple(node) for node in cl)


class Bisimulation(QuotientGraph):
    def __init__(self, dataset, depth=0):
        self.dataset = dataset
        self.depth = depth

    def preprocess(self, multigraph):
        digraph = nx.DiGraph()

        for s, o, data in multigraph.edges(data=True):
            p = data["label"]

            if not digraph.has_node(s):
                digraph.add_node(s, **{"label": self.dataset.id_to_entity[s]})
            if not digraph.has_node(o):
                digraph.add_node(o, **{"label": self.dataset.id_to_entity[o]})
            po_node = (p, o)
            if not digraph.has_node(po_node):
                po_label = f"{p}_{self.dataset.id_to_entity[o]}"
                digraph.add_node(po_node, **{"label": po_label})

            digraph.add_edge(s, po_node)

        return digraph

    def summarize(self, entity, triples):
        subgraph = self.dataset.get_subgraph(entity, triples=triples, depth=self.depth)
        # self.plot(subgraph, filename="subgraph")
        digraph = self.preprocess(subgraph)
        # self.plot(digraph, filename="digraph")
        partition = self.dataset.get_equivalence_classes(subgraph)

        for node in digraph.nodes():
            if is_tuple(node):
                partition.append(frozenset({node}))

        bisimulation = compute_maximum_bisimulation(digraph, partition)
        bisimulation = [frozenset(cl) for cl in bisimulation]

        bisimulation = [cl for cl in bisimulation if not contain_tuples(cl)]

        quotient_graph = self.build_quotient_graph(subgraph, bisimulation, all)
        # self.plot(quotient_graph, filename="bisimulation")

        q_triples = [(s, l, o) for s, o, l in quotient_graph.edges(data="id")]
        filtered_q_triples = []
        entities = [s for s, _, _ in triples]
        entities.extend([o for _, _, o in triples])

        entity_q_triples = [
            (h_part, relation, t_part)
            for (h_part, relation, t_part) in q_triples
            if entity in h_part or entity in t_part
        ]

        for q_triple in entity_q_triples:
            s_part, p, o_part = q_triple
            if any([s in entities for s in s_part]) and any([o in entities for o in o_part]):
                s_part = frozenset([s for s in s_part if s in entities])
                o_part = frozenset([o for o in o_part if o in entities])

                filtered_q_triples.append((s_part, p, o_part))

        self.set_quotient_triple_to_triples(filtered_q_triples, triples)
        
        return list(self.quotient_triple_to_triples.keys())
