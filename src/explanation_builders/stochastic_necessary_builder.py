import itertools
import random
from typing import Tuple, Any

from .explanation_builder import NecessaryExplanationBuilder
from ..data import Dataset
from ..relevance_engines import PostTrainingEngine
from ..link_prediction.models import Model

DEAFAULT_XSI_THRESHOLD = 5


class StochasticNecessaryExplanationBuilder(NecessaryExplanationBuilder):
    """
    The StochasticNecessaryExplanationBuilder object guides the search for necessary rules with a probabilistic policy
    """

    def __init__(
        self,
        model: Model,
        dataset: Dataset,
        hyperparameters: dict,
        triple_to_explain: Tuple[Any, Any, Any],
        perspective: str,
        relevance_threshold: float = None,
        max_explanation_length: int = -1,
    ):
        """StochasticNecessaryExplanationBuilder object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        :param hyperparameters: the hyperparameters of the model and of its optimization process
        :param triple_to_explain: the predicted triple to explain
        :param perspective: the explanation perspective, either "head" or "tail"
        :param max_explanation_length: the maximum number of facts to include in the explanation to extract
        """

        super().__init__(
            model=model,
            dataset=dataset,
            triple_to_explain=triple_to_explain,
            perspective=perspective,
            max_explanation_length=max_explanation_length,
        )

        self.xsi = (
            relevance_threshold
            if relevance_threshold is not None
            else DEAFAULT_XSI_THRESHOLD
        )
        self.window_size = 10
        self.engine = PostTrainingEngine(
            model=model, dataset=dataset, hyperparameters=hyperparameters
        )

    def build_explanations(self, triples_to_remove: list, top_k: int = 10):
        rule_to_relevance = []

        triple_to_relevance = self.singleton_rules(triples_to_remove)

        rule_to_relevance += [
            ([x], y)
            for (x, y) in sorted(
                triple_to_relevance.items(), key=lambda x: x[1], reverse=True
            )
        ]

        triples_number = len(triple_to_relevance)

        _, best = rule_to_relevance[0]
        if best > self.xsi:
            return rule_to_relevance

        rule_length = 2
        while rule_length <= triples_number and rule_length <= self.length_cap:
            current_rule_to_relevance = self.compound_rules(
                triples_to_remove=triples_to_remove,
                length=rule_length,
                triple_to_relevance=triple_to_relevance,
            )
            current_rule_to_relevance = sorted(
                current_rule_to_relevance.items(), key=lambda x: x[1], reverse=True
            )
            rule_to_relevance += current_rule_to_relevance

            _, current_best = current_rule_to_relevance[0]
            if current_best > best:
                best = current_best
            if best > self.xsi:
                break

            rule_length += 1

        return sorted(rule_to_relevance, key=lambda x: x[1], reverse=True)[:top_k]

    def singleton_rules(self, triples_to_remove: list):
        triple_to_relevance = {}

        for i, triple_to_remove in enumerate(triples_to_remove):
            relevance = self.compute_rule_relevance(([triple_to_remove]))
            triple_to_relevance[triple_to_remove] = relevance
            print(
                f"\n\tRelevance for triple {i + 1} on {len(triples_to_remove)}: "
                f"{self.dataset.printable_triple(triple_to_remove)} = {relevance:.3f}"
            )
        return triple_to_relevance

    def compound_rules(
        self, triples_to_remove: list, length: int, triple_to_relevance: dict
    ):
        rules = itertools.combinations(triples_to_remove, length)
        rule_to_pre_score = [
            (x, self.rule_pre_score(x, triple_to_relevance)) for x in rules
        ]
        rule_to_pre_score = sorted(rule_to_pre_score, key=lambda x: x[1], reverse=True)

        rule_to_relevance = {}

        terminate = False
        best = -1e6

        sliding_window = [None for _ in range(self.window_size)]

        i = 0
        for rule, _ in rule_to_pre_score:
            if terminate:
                break

            relevance = self.compute_rule_relevance(rule)
            rule_to_relevance[rule] = relevance
            print(
                f"\nRelevance for rule: {self.dataset.printable_nple(rule)} = {relevance:.3f}"
            )
            sliding_window[i % self.window_size] = relevance

            if relevance > self.xsi:
                return rule_to_relevance
            elif relevance >= best:
                best = relevance
                i += 1
            elif i < self.window_size:
                i += 1
            else:
                avg_window_relevance = sum(sliding_window) / self.window_size
                terminate_threshold = avg_window_relevance / best
                random_value = random.random()
                terminate = random_value > terminate_threshold

                print()
                print(f"\tRelevance {relevance:.3f}")
                print(f"\tAverage window relevance {avg_window_relevance:.3f}")
                print(f"\tMax relevance seen so far {best:.3f}")
                print(f"\tTerminate threshold: {terminate_threshold:.3f}")
                print(f"\tRandom value: {random_value:.3f}")
                print(f"\tTerminate: {str(terminate)}")
                i += 1

        return rule_to_relevance

    def compute_rule_relevance(self, rule: list):
        (
            relevance,
            _
        ) = self.engine.removal_relevance(
            triple_to_explain=self.triple_to_explain,
            perspective=self.perspective,
            triples_to_remove=rule,
        )

        return relevance

    def rule_pre_score(self, rule, triple_to_relevance):
        return sum([triple_to_relevance[x] for x in rule])
