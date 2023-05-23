from typing import Tuple, Any
from ..data import Dataset

TOPOLOGY_PREFILTER = "topology_based"
WEIGHTED_TOPOLOGY_PREFILTER = "weighted_topology_based"
TYPE_PREFILTER = "type_based"
NO_PREFILTER = "none"


class PreFilter:

    """
    The PreFilter object is the manager of the prefilter process.
    It implements the prefiltering pipeline.
    """

    def __init__(self, dataset: Dataset):
        """PreFilter constructor.

        :param dataset: the dataset used to train the model
        """
        self.dataset = dataset

    def most_promising_triples_for(
        self,
        triple_to_explain: Tuple[Any, Any, Any],
        perspective: str,
        top_k=50,
        verbose=True,
    ):
        """This method extracts the top k promising triples for interpreting the triple to explain,
        from the perspective of either its head or its tail (that is, either featuring its head or its tail).

        :param triple_to_explain: the triple to explain
        :param perspective: a string conveying the explanation perspective. It can be either "head" or "tail":
            - if "head", find the most promising triples featuring the head of the triple to explain
            - if "tail", find the most promising triples featuring the tail of the triple to explain
        :param top_k: the number of top promising triples to extract.
        :return: the sorted list of the most promising triples.
        """
        if verbose:
            print(
                f"Extracting promising triples for {self.dataset.printable_triple(triple_to_explain)}"
            )
