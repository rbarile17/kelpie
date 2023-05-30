from typing import Tuple, Any
from ..data import Dataset
from ..link_prediction.models import Model


class SufficientExplanationBuilder:
    """
    The SufficientExplanationBuilder object guides the search for sufficient explanations.
    """

    def __init__(
        self,
        model: Model,
        dataset: Dataset,
        triple_to_explain: Tuple[Any, Any, Any],
        perspective: str,
        num_entities_to_convert: int,
        max_explanation_length: int,
    ):
        """SufficientExplanationBuilder object constructor."""
        self.model = model
        self.dataset = dataset
        self.triple_to_explain = triple_to_explain
        head, _, tail = triple_to_explain

        self.perspective = perspective
        self.perspective_entity = head if perspective == "head" else tail

        self.num_entities_to_convert = num_entities_to_convert
        self.length_cap = max_explanation_length

    def build_explanations(self, triples_to_add: list, top_k: int = 10):
        pass


class NecessaryExplanationBuilder:
    """
    The NecessaryExplanationBuilder object guides the search for necessary explanations.
    """

    def __init__(
        self,
        model: Model,
        dataset: Dataset,
        triple_to_explain: Tuple[Any, Any, Any],
        perspective: str,
        max_explanation_length: int,
    ):
        """NecessaryExplanationBuilder object constructor."""
        self.model = model
        self.dataset = dataset
        self.triple_to_explain = triple_to_explain
        head, _, tail = triple_to_explain

        self.perspective = perspective
        self.perspective_entity = head if perspective == "head" else tail

        self.length_cap = max_explanation_length

    def build_explanations(self, triples_to_remove: list, top_k: int = 10):
        pass
