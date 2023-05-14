from typing import Tuple, Any
from .explanation_builder import NecessaryExplanationBuilder
from ..data import Dataset
from ..relevance_engines import CriageEngine
from ..link_prediction.models import Model


class CriageNecessaryExplanationBuilder(NecessaryExplanationBuilder):

    """
    The CriageNecessaryExplanationBuilder object guides the search for necessary facts to remove for Criage
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
        CriageNecessaryExplanationBuilder object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        :param hyperparameters: the hyperparameters of the model and of its optimization process
        :param perspective
        """

        super().__init__(model, dataset, triple_to_explain, perspective, 1)

        self.engine = CriageEngine(
            model=model, dataset=dataset, hyperparameters=hyperparameters
        )

    def build_explanations(self, triples_to_remove: list, top_k: int = 10):
        rule_2_relevance = {}

        (head_to_explain, _, tail_to_explain) = self.triple_to_explain

        for i, triple_to_remove in enumerate(triples_to_remove):
            print(
                "\n\tComputing relevance for triple "
                + str(i)
                + " on "
                + str(len(triples_to_remove))
                + ": "
                + self.dataset.printable_triple(triple_to_remove)
            )

            tail_to_remove = triple_to_remove[2]

            if tail_to_remove == head_to_explain:
                perspective = "head"
            elif tail_to_remove == tail_to_explain:
                perspective = "tail"
            else:
                raise ValueError

            relevance = self.engine.removal_relevance(
                triple_to_explain=self.triple_to_explain,
                perspective=perspective,
                triples_to_remove=[triple_to_remove],
            )

            rule_2_relevance[tuple([triple_to_remove])] = relevance

        return sorted(rule_2_relevance.items(), key=lambda x: x[1])[:top_k]
