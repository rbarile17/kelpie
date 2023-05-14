from collections import defaultdict
from typing import Tuple, Any
from .explanation_builder import SufficientExplanationBuilder
from ..data import Dataset
from ..relevance_engines import CriageEngine
from ..link_prediction.models import Model


class CriageSufficientExplanationBuilder(SufficientExplanationBuilder):

    """
    The CriageSufficientExplanationBuilder object guides the search for sufficient rules for Criage
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
        CriageSufficientExplanationBuilder object constructor.

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

        self.engine = CriageEngine(
            model=model, dataset=dataset, hyperparameters=hyperparameters
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

        head_to_explain, rel_to_explain, tail_to_explain = self.triple_to_explain

        # try all rules with length 1 are tested
        for i, triple_to_add in enumerate(triples_to_add):
            head_to_add, rel_to_add, tail_to_add = triple_to_add

            if tail_to_add == tail_to_explain:
                perspective = "tail"
            elif tail_to_add == head_to_explain:
                perspective = "head"
            else:
                raise ValueError

            print(
                "\n\tComputing relevance for triple "
                + str(i)
                + " on "
                + str(len(triples_to_add))
                + ": "
                + self.dataset.printable_triple(triple_to_add)
            )
            rule = tuple([triple_to_add])
            global_relevance = self._compute_relevance_for_rule(
                rule=rule, perspective=perspective
            )
            rule_2_global_relevance[rule] = global_relevance
            print("\tObtained global relevance: " + str(global_relevance))

        # relevance is score increase after addition, so sort by highest to lowest relevance
        return sorted(
            rule_2_global_relevance.items(), key=lambda x: x[1], reverse=True
        )[:top_k]

    def _compute_relevance_for_rule(self, rule: Tuple, perspective: str):
        rule_length = len(rule)
        assert len(rule[0]) == 3
        assert rule_length == 1

        triple_to_add = rule[0]

        rule_2_individual_relevances = defaultdict(lambda: [])
        outlines = []

        if perspective == "head":
            assert triple_to_add[2] == self.triple_to_explain[0]
        else:
            assert triple_to_add[2] == self.triple_to_explain[2]

        for j, entity_to_convert in enumerate(self.entities_to_convert):
            print(
                "\t\tConverting entity "
                + str(j)
                + " on "
                + str(self.num_entities_to_convert)
                + ": "
                + self.dataset.id_to_entity[entity_to_convert]
            )

            if perspective == "head":
                r_triple_to_add = (
                    triple_to_add[0],
                    triple_to_add[1],
                    entity_to_convert,
                )
                r_triple_to_convert = (
                    entity_to_convert,
                    self.triple_to_explain[1],
                    self.triple_to_explain[2],
                )
            else:
                r_triple_to_add = (
                    triple_to_add[0],
                    triple_to_add[1],
                    entity_to_convert,
                )
                r_triple_to_convert = (
                    self.triple_to_explain[0],
                    self.triple_to_explain[1],
                    entity_to_convert,
                )

            # if rule length is 1 try all r_triples_to_add and get their individual relevances
            individual_relevance = self.engine.addition_relevance(
                triple_to_convert=r_triple_to_convert,
                perspective=perspective,
                triples_to_add=[r_triple_to_add],
            )

            rule_2_individual_relevances[rule].append(individual_relevance)

            outlines.append(
                ";".join(self.dataset.labels_triple(self.triple_to_explain))
                + ";"
                + ";".join(self.dataset.labels_triple(r_triple_to_convert))
                + ";"
                + ";".join(self.dataset.labels_triple(triple_to_add))
                + ";"
                + str(individual_relevance)
            )

        # add the rule global relevance to all the outlines that refer to this rule
        global_relevance = self._average(rule_2_individual_relevances[rule])

        return global_relevance
