from typing import Tuple, Any
from pprint import pprint
from .explanation_builder import NecessaryExplanationBuilder
from ..data import Dataset
from ..relevance_engines import DataPoisoningEngine
from ..link_prediction.models import Model, LEARNING_RATE


class DataPoisoningNecessaryExplanationBuilder(NecessaryExplanationBuilder):

    """
    The DataPoisoningNecessaryExplanationBuilder object guides the search for DP necessary rules

    """

    def __init__(
        self,
        model: Model,
        dataset: Dataset,
        hyperparameters: dict,
        triple_to_explain: Tuple[Any, Any, Any],
        perspective: str,
    ):
        """
        DataPoisoningNecessaryExplanationBuilder object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        :param hyperparameters: the hyperparameters of the model and of its optimization process
        :param perspective
        """

        super().__init__(model, dataset, triple_to_explain, perspective, 1)

        self.engine = DataPoisoningEngine(
            model=model,
            dataset=dataset,
            hyperparameters=hyperparameters,
            epsilon=hyperparameters[LEARNING_RATE],
        )

    def build_explanations(self, triples_to_remove: list, top_k: int = 10):
        rule_2_relevance = {}

        for i, triple_to_remove in enumerate(triples_to_remove):
            print(
                "\n\tComputing relevance for triple "
                + str(i)
                + " on "
                + str(len(triples_to_remove))
                + ": "
                + self.dataset.printable_triple(triple_to_remove)
            )

            (
                relevance,
                original_target_entity_score,
                original_target_entity_rank,
                original_removed_triple_score,
                perturbed_removed_triple_score,
            ) = self.engine.removal_relevance(
                triple_to_explain=self.triple_to_explain,
                perspective=self.perspective,
                triples_to_remove=[triple_to_remove],
            )

            rule_2_relevance[tuple([triple_to_remove])] = relevance

        pprint(rule_2_relevance)
        return sorted(rule_2_relevance.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]
