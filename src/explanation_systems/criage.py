from typing import Tuple, Any
from ..data import Dataset
from ..prefilters import CriagePreFilter
from ..relevance_engines import CriageEngine
from ..link_prediction.models import Model
from ..explanation_builders import (
    CriageNecessaryExplanationBuilder,
    CriageSufficientExplanationBuilder,
)


class Criage:
    """
    The Criage object is the overall manager of the Criage explanation process.
    It implements the whole explanation pipeline, requesting the suitable operations to the ExplanationEngines
    and to the entity_similarity modules.
    """

    def __init__(self, model: Model, dataset: Dataset, hyperparameters: dict):
        """
        Criage object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        :param hyperparameters: the hyperparameters of the model and of its optimization process
        """
        self.model = model
        self.dataset = dataset
        self.hyperparameters = hyperparameters

        self.prefilter = CriagePreFilter(dataset=dataset)
        self.engine = CriageEngine(
            model=model, dataset=dataset, hyperparameters=hyperparameters
        )

    def explain_necessary(
        self,
        triple_to_explain: Tuple[Any, Any, Any],
        perspective: str,
        num_promising_triples=-1,
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

        top_promising_triples = self.prefilter.most_promising_triples_for(
            triple_to_explain=triple_to_explain,
            perspective=perspective,
            top_k=num_promising_triples,
        )

        explanation_builder = CriageNecessaryExplanationBuilder(
            model=self.model,
            dataset=self.dataset,
            hyperparameters=self.hyperparameters,
            triple_to_explain=triple_to_explain,
            perspective=perspective,
        )

        rules_with_relevance = explanation_builder.build_explanations(
            triples_to_remove=top_promising_triples
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
        :param num_entities_to_convert: the number of entities to convert to extract
                                        (if they have to be extracted)
        :param entities_to_convert: the entities to convert
                                    (if they are passed instead of having to be extracted)

        :return: a list containing for each relevant n-ple extracted, a couple containing
                                - that relevant n-ple
                                - its value of relevance
        """

        most_promising_triples = self.prefilter.most_promising_triples_for(
            triple_to_explain=triple_to_explain,
            perspective=perspective,
            top_k=num_promising_triples,
        )

        explanation_builder = CriageSufficientExplanationBuilder(
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
