from typing import Tuple, Any
from ..data import Dataset
from ..prefilters import TOPOLOGY_PREFILTER, TYPE_PREFILTER, NO_PREFILTER
from ..prefilters import NoPreFilter, TopologyPreFilter, TypeBasedPreFilter

from ..relevance_engines import DataPoisoningEngine
from ..explanation_builders import (
    DataPoisoningNecessaryExplanationBuilder,
    DataPoisoningSufficientExplanationBuilder,
)
from ..link_prediction.models import Model, LEARNING_RATE


class DataPoisoning:
    """
    The DataPoisoning object is the overall manager of the Data_poisoning explanation process.
    It implements the whole explanation pipeline, requesting the suitable operations to the ExplanationEngines
    and to the entity_similarity modules.
    """

    def __init__(
        self, model: Model, dataset: Dataset, hyperparameters: dict, prefilter_type: str
    ):
        """
        DataPoisoning object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        :param hyperparameters: the hyperparameters of the model and of its optimization process
        :param prefilter_type: the type of prefilter to employ
        """
        self.model = model
        self.dataset = dataset
        self.hyperparameters = hyperparameters

        if prefilter_type == TOPOLOGY_PREFILTER:
            self.prefilter = TopologyPreFilter(model=model, dataset=dataset)
        elif prefilter_type == TYPE_PREFILTER:
            self.prefilter = TypeBasedPreFilter(model=model, dataset=dataset)
        elif prefilter_type == NO_PREFILTER:
            self.prefilter = NoPreFilter(model=model, dataset=dataset)
        else:
            self.prefilter = TopologyPreFilter(model=model, dataset=dataset)
        self.engine = DataPoisoningEngine(
            model=model,
            dataset=dataset,
            hyperparameters=hyperparameters,
            epsilon=hyperparameters[LEARNING_RATE],
        )

    def explain_necessary(
        self,
        triple_to_explain: Tuple[Any, Any, Any],
        perspective: str,
        num_promising_triples=50,
    ):
        """
        This method extracts necessary explanations for a specific triple,
        from the perspective of either its head or its tail.

        :param triple_to_explain: the triple to explain
        :param perspective: a string conveying the perspective of the requested explanations.
                            It can be either "head" or "tail":
                                - if "head", Kelpie answers the question
                                    "given the triple head and relation, why is the triple tail predicted as tail?"
                                - if "tail", Kelpie answers the question
                                    "given the triple relation and tail, why is the triple head predicted as head?"
        :param num_promising_triples: the number of triples relevant to the triple to explain
                                     that must be identified and removed from the entity under analysis
                                     to verify whether they worsen the target prediction or not

        :return: a list containing for each relevant n-ple extracted, a couple containing
                                - that relevant n-ple
                                - its value of relevance

        """

        most_promising_triples = self.prefilter.most_promising_triples_for(
            triple_to_explain=triple_to_explain,
            perspective=perspective,
            top_k=num_promising_triples,
        )

        rule_extractor = DataPoisoningNecessaryExplanationBuilder(
            model=self.model,
            dataset=self.dataset,
            hyperparameters=self.hyperparameters,
            triple_to_explain=triple_to_explain,
            perspective=perspective,
        )

        rules_with_relevance = rule_extractor.build_explanations(
            triples_to_remove=most_promising_triples
        )
        return rules_with_relevance

    def explain_sufficient(
        self,
        triple_to_explain: Tuple[Any, Any, Any],
        perspective: str,
        num_promising_triples=50,
        num_entities_to_convert=10,
        entities_to_convert=None,
    ):
        """
        This method extracts necessary explanations for a specific triple,
        from the perspective of either its head or its tail.

        :param triple_to_explain: the triple to explain
        :param perspective: a string conveying the perspective of the requested explanations.
                            It can be either "head" or "tail":
                                - if "head", Kelpie answers the question
                                    "given the triple head and relation, why is the triple tail predicted as tail?"
                                - if "tail", Kelpie answers the question
                                    "given the triple relation and tail, why is the triple head predicted as head?"
        :param num_promising_triples: the number of triples relevant to the triple to explain
                                     that must be identified and removed from the entity under analysis
                                     to verify whether they worsen the target prediction or not

        :return: a list containing for each relevant n-ple extracted, a couple containing
                                - that relevant n-ple
                                - its value of relevance
        :param num_entities_to_convert: the number of entities to convert to extract
                                        (if they have to be extracted)
        :param entities_to_convert: the entities to convert
                                    (if they are passed instead of having to be extracted)
        """

        most_promising_triples = self.prefilter.most_promising_triples_for(
            triple_to_explain=triple_to_explain,
            perspective=perspective,
            top_k=num_promising_triples,
        )

        explanation_builder = DataPoisoningSufficientExplanationBuilder(
            model=self.model,
            dataset=self.dataset,
            hyperparameters=self.hyperparameters,
            triple_to_explain=triple_to_explain,
            perspective=perspective,
            num_entities_to_convert=num_entities_to_convert,
            entities_to_convert=entities_to_convert,
        )

        explanations_with_relevance = explanation_builder.build_explanations(
            triples_to_add=most_promising_triples, top_k=10
        )
        return explanations_with_relevance, explanation_builder.entities_to_convert
