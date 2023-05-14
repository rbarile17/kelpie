from typing import Tuple, Any
from .data import Dataset

from .prefilters import (
    TYPE_PREFILTER,
    TOPOLOGY_PREFILTER,
    NO_PREFILTER,
    WEIGHTED_TOPOLOGY_PREFILTER,
)
from .prefilters import (
    NoPreFilter,
    TypeBasedPreFilter,
    TopologyPreFilter,
    WeightedTopologyPreFilter,
)
from .relevance_engines import PostTrainingEngine
from .link_prediction.models import Model
from .explanation_builders import (
    StochasticNecessaryExplanationBuilder,
    StochasticSufficientExplanationBuilder,
)


class Kelpie:
    """
    The Kelpie object is the overall manager of the explanation process.
    It implements the whole explanation pipeline, requesting the suitable operations
    to the Pre-Filter, Explanation Builder and Relevance Engine modules.
    """

    DEFAULT_MAX_LENGTH = 4

    def __init__(
        self,
        model: Model,
        dataset: Dataset,
        hyperparameters: dict,
        prefilter_type: str,
        relevance_threshold: float = None,
        max_explanation_length: int = DEFAULT_MAX_LENGTH,
    ):
        """
        Kelpie object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        :param hyperparameters: the hyperparameters of the model and of its optimization process
        :param prefilter_type: the type of prefilter to employ
        :param relevance_threshold: the threshold of relevance that, if exceeded, leads to explanation acceptance
        :param max_explanation_length: the maximum number of facts that the explanations to extract can contain
        """
        self.model = model
        self.dataset = dataset
        self.hyperparameters = hyperparameters
        self.relevance_threshold = relevance_threshold
        self.max_explanation_length = max_explanation_length

        if prefilter_type == TOPOLOGY_PREFILTER:
            self.prefilter = TopologyPreFilter(model=model, dataset=dataset)
        elif prefilter_type == TYPE_PREFILTER:
            self.prefilter = TypeBasedPreFilter(model=model, dataset=dataset)
        elif prefilter_type == NO_PREFILTER:
            self.prefilter = NoPreFilter(model=model, dataset=dataset)
        elif prefilter_type == WEIGHTED_TOPOLOGY_PREFILTER:
            self.prefilter = WeightedTopologyPreFilter(model=model, dataset=dataset)
        else:
            self.prefilter = TopologyPreFilter(model=model, dataset=dataset)

        self.engine = PostTrainingEngine(
            model=model, dataset=dataset, hyperparameters=hyperparameters
        )

    def explain_sufficient(
        self,
        triple_to_explain: Tuple[Any, Any, Any],
        perspective: str,
        num_promising_triples: int = 50,
        num_entities_to_convert: int = 10,
        entities_to_convert: list = None,
    ):
        """
        This method extracts sufficient explanations for a specific triple,
        from the perspective of either its head or its tail.

        :param triple_to_explain: the triple to explain
        :param perspective: a string conveying the perspective of the requested explanations.
                            It can be either "head" or "tail":
                                - if "head", Kelpie answers the question
                                    "given the triple head and relation, why is the triple tail predicted as tail?"
                                - if "tail", Kelpie answers the question
                                    "given the triple relation and tail, why is the triple head predicted as head?"
        :param num_promising_triples: the number of triples relevant to the triple to explain
                                     that must be identified and added to the extracted similar entities
                                     to verify whether they boost the target prediction or not
        :param num_entities_to_convert: the number of entities to convert to extract
                                        (if they have to be extracted)
        :param entities_to_convert: the entities to convert
                                    (if they are passed instead of having to be extracted)

        :return: two lists:
                    the first one contains, for each relevant n-ple extracted, a couple containing
                                - that relevant triple
                                - its value of global relevance across the entities to convert
                    the second one contains the list of entities that the extractor has tried to convert
                        in the sufficient explanation process

        """

        most_promising_triples = self.prefilter.most_promising_triples_for(
            triple_to_explain=triple_to_explain,
            perspective=perspective,
            top_k=num_promising_triples,
        )

        explanation_builder = StochasticSufficientExplanationBuilder(
            model=self.model,
            dataset=self.dataset,
            hyperparameters=self.hyperparameters,
            triple_to_explain=triple_to_explain,
            perspective=perspective,
            num_entities_to_convert=num_entities_to_convert,
            entities_to_convert=entities_to_convert,
            relevance_threshold=self.relevance_threshold,
            max_explanation_length=self.max_explanation_length,
        )

        explanations_with_relevance = explanation_builder.build_explanations(
            triples_to_add=most_promising_triples
        )
        return explanations_with_relevance, explanation_builder.entities_to_convert

    def explain_necessary(
        self,
        triple_to_explain: Tuple[Any, Any, Any],
        perspective: str,
        num_promising_triples: int = 50,
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

        explanation_builder = StochasticNecessaryExplanationBuilder(
            model=self.model,
            dataset=self.dataset,
            hyperparameters=self.hyperparameters,
            triple_to_explain=triple_to_explain,
            perspective=perspective,
            relevance_threshold=self.relevance_threshold,
            max_explanation_length=self.max_explanation_length,
        )

        explanations_with_relevance = explanation_builder.build_explanations(
            triples_to_remove=most_promising_triples
        )
        return explanations_with_relevance
