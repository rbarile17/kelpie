import networkx as nx

from typing import Tuple, Any

from .prefilter import PreFilter
from ..data import Dataset


class TopologyPreFilter(PreFilter):
    """
    The TopologyPreFilter object is a PreFilter that relies on the graph topology
    to extract the most promising triples for an explanation.
    """

    def __init__(self, dataset: Dataset):
        """PostTrainingPreFilter object constructor."""
        super().__init__(dataset)

        self.graph = nx.MultiGraph()
        self.graph.add_nodes_from(list(dataset.id_to_entity.keys()))
        self.graph.add_edges_from(
            [
                (h, t, {"relation_label": dataset.id_to_relation[r]})
                for h, r, t in dataset.training_triples
            ]
        )

        self.entity_to_training_triples = self.dataset.entity_to_training_triples

    def most_promising_triples_for(
        self,
        triple_to_explain: Tuple[Any, Any, Any],
        perspective: str,
        top_k=50,
        verbose=True,
    ):
        """See base class."""
        super().most_promising_triples_for(
            triple_to_explain, perspective, top_k, verbose
        )
        head, _, tail = triple_to_explain

        start_entity, end_entity = (
            (head, tail) if perspective == "head" else (tail, head)
        )

        triple_to_analyze_2_min_path_length = {
            triple_to_analyze: self.analyze_triple(
                (start_entity, end_entity, triple_to_analyze)
            )
            for triple_to_analyze in sorted(
                self.entity_to_training_triples[start_entity],
                key=lambda x: (x[0], x[1], x[2]),
            )
        }

        results = sorted(
            triple_to_analyze_2_min_path_length.items(), key=lambda x: x[1]
        )
        results = [x[0] for x in results]

        return results[:top_k]

    def analyze_triple(self, input):
        (
            start_entity,
            end_entity,
            triple_to_analyze,
        ) = input

        head_to_analyze, _, tail_to_analyze = triple_to_analyze

        if head_to_analyze == start_entity:
            entity_to_analyze = tail_to_analyze
        elif tail_to_analyze == start_entity:
            entity_to_analyze = head_to_analyze
        else:
            raise ValueError("Start entity not found in triple to analyze")

        try:
            return nx.shortest_path_length(
                self.graph,
                entity_to_analyze,
                end_entity,
            )
        except nx.NetworkXNoPath:
            return 1e6
