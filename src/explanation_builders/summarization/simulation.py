from .quotient_graph import QuotientGraph


class Simulation(QuotientGraph):
    def __init__(self, dataset):
        self.dataset = dataset

    def summarize(self, entity, triples):
        subgraph = self.dataset.get_subgraph(entity, triples, 0)
        # self.plot(subgraph, filename="subgraph")
        equivalence_classes = self.dataset.get_equivalence_classes(subgraph)
        quotient_graph = self.build_quotient_graph(subgraph, equivalence_classes, any)
        # self.plot(quotient_graph, filename="simulation")
        q_triples = [(s, l, o) for s, o, l in quotient_graph.edges(data="id")]

        self.set_quotient_triple_to_triples(q_triples, triples)
        return q_triples
