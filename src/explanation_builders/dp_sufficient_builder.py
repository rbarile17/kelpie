import numpy

from collections import defaultdict
from typing import Tuple, Any
from .explanation_builder import SufficientExplanationBuilder
from ..data import Dataset
from ..relevance_engines import DataPoisoningEngine
from ..link_prediction.models import Model, LEARNING_RATE


class DataPoisoningSufficientExplanationBuilder(SufficientExplanationBuilder):

    """
    The DataPoisoningSufficientExplanationBuilder object guides the search for sufficient explanations for DP
    """

    def __init__(
        self,
        model: Model,
        dataset: Dataset,
        hyperparameters: dict,
        triple_to_explain: Tuple[Any, Any, Any],
        perspective: str,
        num_entities_to_convert=10,
        entities_to_convert=None,
    ):
        """
        DataPoisoningSufficientExplanationBuilder object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        :param hyperparameters: the hyperparameters of the model and of its optimization process
        :param triple_to_explain
        :param perspective
        :param num_entities_to_convert
        """

        super().__init__(
            model, dataset, triple_to_explain, perspective, num_entities_to_convert, 1
        )

        self.engine = DataPoisoningEngine(
            model=model,
            dataset=dataset,
            hyperparameters=hyperparameters,
            epsilon=hyperparameters[LEARNING_RATE],
        )

        if entities_to_convert is not None:
            assert len(entities_to_convert) == num_entities_to_convert
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
        rule_2_global_relevance = {}

        # this is an exception: all rules with length 1 are tested
        for i, triple_to_add in enumerate(triples_to_add):
            print(
                "\n\tComputing relevance for triple "
                + str(i)
                + " on "
                + str(len(triples_to_add))
                + ": "
                + self.dataset.printable_triple(triple_to_add)
            )
            rule = tuple([triple_to_add])
            global_relevance = self._compute_relevance_for_rule(rule)
            rule_2_global_relevance[rule] = global_relevance
            print("\tObtained global relevance: " + str(global_relevance))

        return sorted(
            rule_2_global_relevance.items(), key=lambda x: x[1], reverse=True
        )[:top_k]

    def _compute_relevance_for_rule(self, rule: Tuple):
        rule_length = len(rule)
        assert len(rule[0]) == 3
        assert rule_length == 1

        triple_to_add = rule[0]

        rule_2_individual_relevances = defaultdict(lambda: [])
        outlines = []

        for j, entity_to_convert in enumerate(self.entities_to_convert):
            print(
                "\t\tConverting entity "
                + str(j)
                + " on "
                + str(self.num_entities_to_convert)
                + ": "
                + self.dataset.id_to_entity[entity_to_convert]
            )

            r_nple_to_add = Dataset.replace_entity_in_triples(
                triples=rule,
                old_entity=self.perspective_entity,
                new_entity=entity_to_convert,
            )
            r_triple_to_convert = Dataset.replace_entity_in_triple(
                self.triple_to_explain, self.perspective_entity, entity_to_convert
            )

            # if rule length is 1 try all r_triples_to_add and get their individual relevances
            (
                individual_relevance,
                original_target_entity_score,
                original_target_entity_rank,
                original_added_triple_score,
                perturbed_added_triple_score,
            ) = self.engine.addition_relevance(
                triple_to_convert=r_triple_to_convert,
                perspective=self.perspective,
                triples_to_add=r_nple_to_add,
            )

            rule_2_individual_relevances[rule].append(individual_relevance)

        # add the rule global relevance to all the outlines that refer to this rule
        global_relevance = sum(rule_2_individual_relevances[rule])
        global_relevance /= len(rule_2_individual_relevances[rule])

        return global_relevance

    def _preliminary_rule_score(self, rule, triple_2_relevance):
        return numpy.sum([triple_2_relevance[x] for x in rule])
