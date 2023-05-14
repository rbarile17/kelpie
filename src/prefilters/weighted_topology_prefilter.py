import networkx as nx
import pandas as pd

from ast import literal_eval
from collections import defaultdict
from typing import Tuple, Any

from .. import DBPEDIA50_REASONED_PATH

from .prefilter import PreFilter
from ..dataset import Dataset
from ..link_prediction.models import Model
from ..utils import jaccard_similarity


class WeightedTopologyPreFilter(PreFilter):
    """
    The TopologyPreFilter object is a PreFilter that relies on the graph topology
    to extract the most promising samples for an explanation.
    """

    def __init__(self, model: Model, dataset: Dataset):
        """
        PostTrainingPreFilter object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        """
        super().__init__(model, dataset)

        self.graph = nx.MultiGraph()
        self.graph.add_nodes_from(list(dataset.entity_id_2_name.keys()))
        self.graph.add_edges_from(
            [
                (h, t, {"relation_label": dataset.get_name_for_relation_id(r)})
                for h, r, t in dataset.train_samples
            ]
        )

        self.entities_semantic = pd.read_csv(
            DBPEDIA50_REASONED_PATH / "entities.csv",
            converters={"classes": literal_eval},
        )

        self.entities_semantic = self.entities_semantic.set_index("entity")[
            ["classes"]
        ].to_dict(orient="index")

        self.entity_id_2_train_samples = defaultdict(list)

        for head, relation, tail in dataset.train_samples:
            self.entity_id_2_train_samples[head].append((head, relation, tail))
            self.entity_id_2_train_samples[tail].append((head, relation, tail))

    def most_promising_samples_for(
        self,
        sample_to_explain: Tuple[Any, Any, Any],
        perspective: str,
        top_k=50,
        verbose=True,
    ):
        """See base class."""
        head, relation, tail = sample_to_explain
        self.relation_to_explain = relation

        if verbose:
            print(
                f"Extracting promising facts for {self.dataset.printable_sample(sample_to_explain)}"
            )

        start_entity, end_entity = (
            (head, tail) if perspective == "head" else (tail, head)
        )

        sample_to_analyze_2_min_path_length = {
            sample_to_analyze: self.analyze_sample(
                (start_entity, end_entity, sample_to_analyze)
            )
            for sample_to_analyze in self.entity_id_2_train_samples[start_entity]
        }

        results = sorted(
            sample_to_analyze_2_min_path_length.items(), key=lambda x: x[1]
        )
        results = [x[0] for x in results]

        return results[:top_k]

    def semantic_score_entities(self, entity1, entity2, edge_data):
        entity1 = self.dataset.get_name_for_entity_id(entity1)
        entity2 = self.dataset.get_name_for_entity_id(entity2)

        return 1 - jaccard_similarity(
            self.entities_semantic[entity1]["classes"],
            self.entities_semantic[entity2]["classes"],
        )

    def analyze_sample(self, input):
        (
            start_entity,
            end_entity,
            sample_to_analyze,
        ) = input

        head_to_analyze, _, tail_to_analyze = sample_to_analyze

        if head_to_analyze == start_entity:
            entity_to_analyze = tail_to_analyze
        elif tail_to_analyze == start_entity:
            entity_to_analyze = head_to_analyze
        else:
            raise ValueError("Start entity not found in sample to analyze")

        try:
            return nx.shortest_path_length(
                self.graph,
                entity_to_analyze,
                end_entity,
                weight=self.semantic_score_entities,
            )
        except nx.NetworkXNoPath:
            return 1e6
