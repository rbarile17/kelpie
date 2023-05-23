import threading
import numpy as np

from multiprocessing.pool import ThreadPool as Pool
from collections import defaultdict
from typing import Tuple, Any

from .prefilter import PreFilter
from ..config import MAX_PROCESSES
from ..data import Dataset


class TypeBasedPreFilter(PreFilter):
    """
    The TypeBasedPreFilter object is a PreFilter that relies on the entity types
    to extract the most promising triples for an explanation.
    """

    def __init__(self, dataset: Dataset):
        """TypeBasedPreFilter object constructor."""
        super().__init__(dataset)

        self.entity_to_training_triples = self.dataset.entity_to_training_triples
        self.entity_to_relation_vector = defaultdict(
            lambda: np.zeros(
                (self.dataset.num_entities, self.dataset.num_relations * 2)
            )
        )

        self.threadLock = threading.Lock()
        self.counter = 0
        self.thread_pool = Pool(processes=MAX_PROCESSES)

        for head, relation, tail in dataset.training_triples:
            self.entity_to_relation_vector[head][relation] += 1
            self.entity_to_relation_vector[tail][
                relation + self.dataset.num_relations
            ] += 1

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

        self.counter = 0

        head, _, tail = triple_to_explain

        if perspective == "head":
            triples_featuring_head = self.entity_to_training_triples[head]

            worker_processes_inputs = [
                (
                    len(triples_featuring_head),
                    triple_to_explain,
                    x,
                    perspective,
                    verbose,
                )
                for x in triples_featuring_head
            ]
            results = self.thread_pool.map(
                self._analyze_triple, worker_processes_inputs
            )

            triple_featuring_head_2_promisingness = {}

            for i in range(len(triples_featuring_head)):
                triple_featuring_head_2_promisingness[
                    triples_featuring_head[i]
                ] = results[i]

            triples_featuring_head_with_promisingness = sorted(
                triple_featuring_head_2_promisingness.items(),
                reverse=True,
                key=lambda x: x[1],
            )
            sorted_promising_triples = [
                x[0] for x in triples_featuring_head_with_promisingness
            ]

        else:
            triples_featuring_tail = self.entity_to_training_triples[tail]

            worker_processes_inputs = [
                (
                    len(triples_featuring_tail),
                    triple_to_explain,
                    x,
                    perspective,
                    verbose,
                )
                for x in triples_featuring_tail
            ]
            results = self.thread_pool.map(
                self._analyze_triple, worker_processes_inputs
            )

            triple_featuring_tail_2_promisingness = {}

            for i in range(len(triples_featuring_tail)):
                triple_featuring_tail_2_promisingness[
                    triples_featuring_tail[i]
                ] = results[i]

            triples_featuring_tail_with_promisingness = sorted(
                triple_featuring_tail_2_promisingness.items(),
                reverse=True,
                key=lambda x: x[1],
            )
            sorted_promising_triples = [
                x[0] for x in triples_featuring_tail_with_promisingness
            ]

        return sorted_promising_triples[:top_k]

    def _analyze_triple(self, input_data):
        (
            all_triples_number,
            triple_to_explain,
            triple_to_analyze,
            perspective,
            verbose,
        ) = input_data

        with self.threadLock:
            self.counter += 1
            i = self.counter

        if verbose:
            print(
                f"\tAnalyzing triple {i} on {all_triples_number}: {self.dataset.printable_triple(triple_to_analyze)}"
            )

        head_to_explain, _, tail_to_explain = triple_to_explain
        head_to_analyze, _, tail_to_analyze = triple_to_analyze

        promisingness = -1
        if perspective == "head":
            if head_to_explain == head_to_analyze:
                promisingness = self._cosine_similarity(
                    tail_to_explain, tail_to_analyze
                )
            else:
                assert head_to_explain == tail_to_analyze
                promisingness = self._cosine_similarity(
                    tail_to_explain, head_to_analyze
                )

        elif perspective == "tail":
            if tail_to_explain == tail_to_analyze:
                promisingness = self._cosine_similarity(
                    head_to_explain, head_to_analyze
                )
            else:
                assert tail_to_explain == head_to_analyze
                promisingness = self._cosine_similarity(
                    head_to_explain, tail_to_analyze
                )

        return promisingness

    def _cosine_similarity(self, entity1_id, entity2_id):
        entity1_vector = self.entity_to_relation_vector[entity1_id]
        entity2_vector = self.entity_to_relation_vector[entity2_id]
        return np.inner(entity1_vector, entity2_vector) / (
            np.linalg.norm(entity1_vector) * np.linalg.norm(entity2_vector)
        )
