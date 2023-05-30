import itertools
import random
from collections import defaultdict
from typing import Tuple, Any

from .explanation_builder import SufficientExplanationBuilder
from ..data import Dataset
from ..relevance_engines import PostTrainingEngine
from ..link_prediction.models import Model

DEAFAULT_XSI_THRESHOLD = 0.9


class StochasticSufficientExplanationBuilder(SufficientExplanationBuilder):
    """
    The StochasticSufficientExplanationBuilder object guides the search for sufficient explanations with a probabilistic policy
    """

    def __init__(
        self,
        model: Model,
        dataset: Dataset,
        hyperparameters: dict,
        triple_to_explain: Tuple[Any, Any, Any],
        perspective: str,
        num_entities_to_convert: int = 10,
        entities_to_convert: list = None,
        relevance_threshold: float = None,
        max_explanation_length: int = -1,
    ):
        """
        StochasticSufficientExplanationBuilder object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        :param hyperparameters: the hyperparameters of the model and of its optimization process
        :param triple_to_explain: the predicted triple to explain
        :param perspective: the explanation perspective, either "head" or "tail"
        :param num_entities_to_convert
        :param max_explanation_length: the maximum number of facts to include in the explanation to extract
        """

        super().__init__(
            model=model,
            dataset=dataset,
            triple_to_explain=triple_to_explain,
            perspective=perspective,
            num_entities_to_convert=num_entities_to_convert,
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

        if entities_to_convert is not None:
            self.entities_to_convert = entities_to_convert
        else:
            self.entities_to_convert = self.engine.extract_entities_for(
                model=self.model,
                dataset=self.dataset,
                triple=triple_to_explain,
                perspective=perspective,
                k=num_entities_to_convert,
                degree_cap=200,
            )

    def build_explanations(self, triples_to_add: list, top_k: int = 10):
        rule_to_relevance = []

        # get relevance for rules with length 1 (that is, triples)
        triple_to_relevance = self.singleton_rules(triples_to_add=triples_to_add)
        triples_number = len(triple_to_relevance)
        rule_to_relevance += [
            ([x], y)
            for (x, y) in sorted(
                triple_to_relevance.items(), key=lambda x: x[1], reverse=True
            )
        ]

        _, best = rule_to_relevance[0]
        if best > self.xsi:
            return rule_to_relevance

        rule_length = 2
        while rule_length <= triples_number and rule_length <= self.length_cap:
            current_rule_to_relevance = self.compound_rules(
                triples_to_add=triples_to_add,
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

    def singleton_rules(self, triples_to_add: list):
        triple_to_relevance = {}

        for i, triple_to_add in enumerate(triples_to_add):
            global_relevance = self.compute_rule_relevance(tuple([triple_to_add]))
            triple_to_relevance[triple_to_add] = global_relevance
            print(
                f"Relevance for triple {i + 1} on {len(triples_to_add)}: "
                f"{self.dataset.printable_triple(triple_to_add)} = {global_relevance:.3f}"
            )

        return triple_to_relevance

    def compound_rules(
        self, triples_to_add: list, length: int, triple_to_relevance: dict
    ):
        rules = itertools.combinations(triples_to_add, length)
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
                f"Relevance for rule: {self.dataset.printable_nple(rule)} = {relevance:.3f}"
            )

            # put the obtained relevance in the window
            sliding_window[i % self.window_size] = relevance

            # early termination
            if relevance > self.xsi:
                return rule_to_relevance
            elif best is None or relevance >= best:
                best = relevance
                i += 1
            elif i < self.window_size:
                i += 1
            else:
                avg_window_relevance = sum(sliding_window) / self.window_size
                terminate_threshold = avg_window_relevance / best
                random_value = random.random()
                terminate = random_value > terminate_threshold  # termination condition

                print()
                print(f"\tRelevance {relevance:.3f}")
                print(f"\tAverage window relevance {avg_window_relevance:.3f}")
                print(f"\tMax relevance seen so far {best:.3f}")
                print(f"\tTerminate threshold: {terminate_threshold:.3f}")
                print(f"\tRandom value: {random_value:.3f}")
                print(f"\tTerminate: {str(terminate)}")
                i += 1

        return rule_to_relevance

    def compute_rule_relevance(self, rule: Tuple):
        rule_to_relevance = defaultdict(list)

        for i, entity_to_convert in enumerate(self.entities_to_convert):
            print(
                f"\tConverting entity {str(i)} on {str(len(self.entities_to_convert))}: "
                f"{self.dataset.id_to_entity[entity_to_convert]}"
            )

            converted_rule = Dataset.replace_entity_in_triples(
                triples=rule,
                old_entity=self.perspective_entity,
                new_entity=entity_to_convert,
            )
            converted_triple_to_explain = Dataset.replace_entity_in_triple(
                self.triple_to_explain,
                self.perspective_entity,
                entity_to_convert,
            )

            (individual_relevance, _) = self.engine.addition_relevance(
                triple_to_convert=converted_triple_to_explain,
                perspective=self.perspective,
                triples_to_add=converted_rule,
            )

            rule_to_relevance[rule].append(individual_relevance)

        return sum(rule_to_relevance[rule]) / len(rule_to_relevance[rule])

    def rule_pre_score(self, rule, triple_to_relevance):
        return sum([triple_to_relevance[triple] for triple in rule])
