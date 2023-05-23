from collections import defaultdict
from typing import Tuple, Any

from .prefilter import PreFilter
from ..data import Dataset


class CriagePreFilter(PreFilter):
    """The CriagePreFilter object is a PreFilter that just returns all the triples
    that have, as a tail, either the head or the tail of the triple to explain.
    """

    def __init__(self, dataset: Dataset):
        """CriagePreFilter object constructor."""
        super().__init__(dataset)

        self.tail_to_training_triples = defaultdict(list)

        for h, r, t in dataset.training_triples:
            self.tail_to_training_triples[t].append((h, r, t))

    def most_promising_triples_for(
        self,
        triple_to_explain: Tuple[Any, Any, Any],
        perspective: str,
        top_k=50,
        verbose=True,
    ):
        """See base class.
        This method returns all training triples that have, as a tail,
        either the head or the tail of the triple to explain.

        :param perspective: not used in Criage
        :param verbose: not used in Criage
        """
        super().most_promising_triples_for(
            triple_to_explain, perspective, top_k, verbose
        )
        head, _, tail = triple_to_explain

        tail_as_tail_triples = sorted(
            self.tail_to_training_triples.get(tail, []),
            key=lambda x: (x[0], x[1], x[2]),
        )
        head_as_tail_triples = sorted(
            self.tail_to_training_triples.get(head, []),
            key=lambda x: (x[0], x[1], x[2]),
        )

        if top_k == -1:
            return tail_as_tail_triples + head_as_tail_triples
        return tail_as_tail_triples[:top_k] + head_as_tail_triples[:top_k]
