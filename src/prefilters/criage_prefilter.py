from collections import defaultdict
from typing import Tuple, Any

from .prefilter import PreFilter
from ..data import Dataset
from ..link_prediction.models import Model


class CriagePreFilter(PreFilter):
    """
    The CriagePreFilter object is a PreFilter that just returns all the triples
    that have as tail the same tail as the triple to explain
    """

    def __init__(self, model: Model, dataset: Dataset):
        """
        CriagePreFilter object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        """
        super().__init__(model, dataset)

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
        """
        This method returns all training triples that have, as a tail,
        either the head or the tail of the triple to explain.

        :param triple_to_explain: the triple to explain
        :param perspective: not used in Criage
        :param top_k: the number of triples to return.
        :param verbose: not used in Criage
        :return: the first top_k extracted triples.
        """

        # note: perspective and verbose will be ignored

        head, relation, tail = triple_to_explain

        tail_as_tail_triples = []
        if tail in self.tail_to_training_triples:
            tail_as_tail_triples = self.tail_to_training_triples[tail]

        head_as_tail_triples = []
        if head in self.tail_to_training_triples:
            head_as_tail_triples = self.tail_to_training_triples[head]

        tail_as_tail_triples = sorted(
            tail_as_tail_triples, key=lambda x: (x[0], x[1], x[2])
        )
        head_as_tail_triples = sorted(
            head_as_tail_triples, key=lambda x: (x[0], x[1], x[2])
        )

        if top_k == -1:
            return tail_as_tail_triples + head_as_tail_triples
        else:
            return tail_as_tail_triples[:top_k] + head_as_tail_triples[:top_k]
