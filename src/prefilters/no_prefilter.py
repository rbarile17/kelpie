from typing import Tuple, Any

from .prefilter import PreFilter
from ..data import Dataset


class NoPreFilter(PreFilter):
    """The NoPreFilter object is a fake PreFilter that does not filter away any unpromising facts."""

    def __init__(self, dataset: Dataset):
        """
        NoPreFilter object constructor."""
        super().__init__(dataset)

        self.entity_to_training_triples = self.dataset.entity_to_training_triples

    def most_promising_triples_for(
        self,
        triple_to_explain: Tuple[Any, Any, Any],
        perspective: str,
        top_k=-1,
        verbose=True,
    ):
        """See base class."""
        super().most_promising_triples_for(
            triple_to_explain, perspective, top_k, verbose
        )
        head, _, tail = triple_to_explain

        start_entity = head if perspective == "head" else tail
        return self.entity_to_training_triples[start_entity]
