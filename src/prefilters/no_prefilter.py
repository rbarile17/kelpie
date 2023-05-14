from typing import Tuple, Any

from .prefilter import PreFilter
from ..data import Dataset
from ..link_prediction.models import Model


class NoPreFilter(PreFilter):
    """
    The NoPreFilter object is a fake PreFilter that does not filter away any unpromising facts .
    """

    def __init__(self, model: Model, dataset: Dataset):
        """
        NoPreFilter object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        """
        super().__init__(model, dataset)

        self.entity_id_2_training_triples = self.dataset.entity_to_training_triples

    def most_promising_triples_for(
        self,
        triple_to_explain: Tuple[Any, Any, Any],
        perspective: str,
        top_k=-1,  # not used
        verbose=True,
    ):
        """
        This method extracts the top k promising triples for interpreting the triple to explain,
        from the perspective of either its head or its tail (that is, either featuring its head or its tail).

        :param triple_to_explain: the triple to explain
        :param perspective: a string conveying the explanation perspective. It can be either "head" or "tail":
                                - if "head", find the most promising triples featuring the head of the triple to explain
                                - if "tail", find the most promising triples featuring the tail of the triple to explain
        :param top_k: the number of top promising triples to extract.
        :return: the sorted list of the k most promising triples.
        """

        head, relation, tail = triple_to_explain

        if verbose:
            print(
                "Extracting promising facts for"
                + self.dataset.printable_triple(triple_to_explain)
            )

        start_entity, end_entity = (
            (head, tail) if perspective == "head" else (tail, head)
        )
        triples_featuring_start_entity = self.entity_id_2_training_triples[start_entity]

        return triples_featuring_start_entity
