import networkx as nx
import pandas as pd

from ast import literal_eval
from typing import Tuple, Any

from .. import DBPEDIA50_REASONED_PATH

from .prefilter import PreFilter
from ..data import Dataset
from ..utils import jaccard_similarity


class WeightedTopologyPreFilter(PreFilter):
    """
    The TopologyPreFilter object is a PreFilter that relies on the graph topology
    to extract the most promising triples for an explanation.
    """

    def __init__(self, dataset: Dataset):
        """PostTrainingPreFilter object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        """
        super().__init__(dataset)

        self.graph = nx.MultiGraph()
        self.graph.add_nodes_from(list(dataset.id_to_entity.keys()))
        self.graph.add_edges_from(
            [
                (h, t, {"relation_label": dataset.id_to_relation[r]})
                for h, r, t in dataset.training_triples
            ]
        )

        self.entities_semantic = pd.read_csv(
            DBPEDIA50_REASONED_PATH / "entities.csv",
            converters={"classes": literal_eval},
        )

        self.entities_semantic = self.entities_semantic.set_index("entity")[
            ["classes"]
        ].to_dict(orient="index")

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

        head, relation, tail = triple_to_explain
        self.relation_to_explain = relation

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

    def semantic_score_entities(self, entity1, entity2, edge_data):
        entity1 = self.dataset.id_to_entity[entity1]
        entity2 = self.dataset.id_to_entity[entity2]

        return 1 - jaccard_similarity(
            self.entities_semantic[entity1]["classes"],
            self.entities_semantic[entity2]["classes"],
        )

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
                weight=self.semantic_score_entities,
            )
        except nx.NetworkXNoPath:
            return 1e6
