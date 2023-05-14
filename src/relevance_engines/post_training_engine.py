import math
import time
import numpy
import torch

from typing import Tuple, Any
from collections import OrderedDict

from .engine import ExplanationEngine

from ..data import Dataset, KelpieDataset
from ..link_prediction.models import Model, KelpieModel, ComplEx, ConvE, TransE, TuckER
from ..link_prediction.optimization import (
    KelpieBCEOptimizer,
    KelpieMultiClassNLLOptimizer,
    KelpiePairwiseRankingOptimizer,
)


class PostTrainingEngine(ExplanationEngine):
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def __init__(self, model: Model, dataset: Dataset, hyperparameters: dict):
        """
        PostTrainingEngine constructor.

        :param model: the trained Model to explain the behaviour of. This can NOT be a KelpieModel.
        :param dataset: the Dataset used to train the model
        :param hyperparameters: dict containing all the hyperparameters necessary for running the post-training
                                (for both the model and the optimizer)
        """

        ExplanationEngine.__init__(
            self, model=model, dataset=dataset, hyperparameters=hyperparameters
        )

        if isinstance(self.model, ComplEx):
            self.kelpie_optimizer_class = KelpieMultiClassNLLOptimizer
        elif isinstance(self.model, ConvE):
            self.kelpie_optimizer_class = KelpieBCEOptimizer
        elif isinstance(self.model, TransE):
            self.kelpie_optimizer_class = KelpiePairwiseRankingOptimizer
        else:
            self.kelpie_optimizer_class = KelpieMultiClassNLLOptimizer

        if isinstance(model, KelpieModel):
            raise Exception(
                "The model passed to the PostTrainingEngine is already a post-trainable KelpieModel."
            )

        # these data structures are used store permanently, for any fact:
        #   - the score
        #   - the score obtained by the best scoring tail (in "head" perspective) or head (in "tail" perspective)
        #   - the rank obtained by the target tail (in "head" perspective) or head (in "tail" perspective) score)
        self._original_model_results = (
            {}
        )  # map original triples to scores and ranks from the original model
        self._base_pt_model_results = (
            {}
        )  # map original triples to scores and ranks from the base post-trained model

        # The kelpie_cache is a simple LRU cache that allows reuse of KelpieDatasets and of base post-training results
        # without need to re-build them from scratch every time.
        self._kelpie_dataset_cache_size = 20
        self._kelpie_dataset_cache = OrderedDict()

    def addition_relevance(
        self,
        triple_to_convert: Tuple[Any, Any, Any],
        perspective: str,
        triples_to_add: list,
    ):
        """
        Given a "triple to convert" (that is, a triple that the model currently does not predict as true,
        and that we want to be predicted as true);
        given the perspective from which to analyze it;
        and given and a list of triples containing the entity to convert and that were not seen in training;
        compute the relevance of the triples to add, that is, an estimate of the effect they would have
        if added (all together) to the perspective entity to improve the prediction of the triple to convert.

        :param triple_to_convert: the triple that we would like the model to predict as "true",
                                  in the form of a tuple (head, relation, tail)
        :param perspective: the perspective from which to explain the fact: it can be either "head" or "tail"
        :param triples_to_add: the list of triples containing the perspective entity
                               that we want to analyze the effect of, if added to the perspective entity

        :return:
        """
        start_time = time.time()

        head_id, relation_id, tail_id = triple_to_convert
        original_entity_to_convert = head_id if perspective == "head" else tail_id

        # check how the original model performs on the original triple to convert
        (
            original_target_entity_score,
            original_best_entity_score,
            original_target_entity_rank,
        ) = self.original_results_for(original_triple_to_predict=triple_to_convert)

        # get from the cache a Kelpie Dataset focused on the original id of the entity to explain,
        # (or create it from scratch if it is not in cache)
        kelpie_dataset = self._get_kelpie_dataset_for(
            original_entity_id=original_entity_to_convert
        )

        kelpie_model_class = self.model.kelpie_model_class()

        kelpie_init_tensor_size = (
            self.model.dimension
            if not isinstance(self.model, TuckER)
            else self.model.entity_dimension
        )
        kelpie_init_tensor = torch.rand(1, kelpie_init_tensor_size)

        # run base post-training to obtain a "clone" of the perspective entity and see how it performs in the triple to convert
        base_kelpie_model = kelpie_model_class(
            model=self.model, dataset=kelpie_dataset, init_tensor=kelpie_init_tensor
        )
        (
            base_pt_target_entity_score,
            base_pt_best_entity_score,
            base_pt_target_entity_rank,
        ) = self.base_post_training_results_for(
            kelpie_model=base_kelpie_model,
            kelpie_dataset=kelpie_dataset,
            original_triple_to_predict=triple_to_convert,
        )

        # run actual post-training by adding the passed triples to the perspective entity and see how it performs in the triple to convert
        pt_kelpie_model = kelpie_model_class(
            model=self.model, dataset=kelpie_dataset, init_tensor=kelpie_init_tensor
        )
        (
            pt_target_entity_score,
            pt_best_entity_score,
            pt_target_entity_rank,
        ) = self.addition_post_training_results_for(
            kelpie_model=pt_kelpie_model,
            kelpie_dataset=kelpie_dataset,
            original_triple_to_predict=triple_to_convert,
            original_triples_to_add=triples_to_add,
        )

        # we want to give higher priority to the facts that, when added, make the score the better (= smaller).
        rank_improvement = base_pt_target_entity_rank - pt_target_entity_rank

        # if the model is a minimizer the smaller pt_target_entity_score is than base_pt_target_entity_score, the more relevant the added fact;
        # if the model is a maximizer, the greater pt_target_entity_score is than base_pt_target_entity_score, the more relevant the added fact
        score_improvement = (
            base_pt_target_entity_score - pt_target_entity_score
            if self.model.is_minimizer()
            else pt_target_entity_score - base_pt_target_entity_score
        )

        relevance = float(rank_improvement + self.sigmoid(score_improvement)) / float(
            base_pt_target_entity_rank
        )

        print("\t\tObtained individual relevance: " + str(relevance) + "\n")

        end_time = time.time()
        execution_time = end_time - start_time
        return (
            relevance,
            original_best_entity_score,
            original_target_entity_score,
            original_target_entity_rank,
            base_pt_best_entity_score,
            base_pt_target_entity_score,
            base_pt_target_entity_rank,
            pt_best_entity_score,
            pt_target_entity_score,
            pt_target_entity_rank,
            execution_time,
        )

    def removal_relevance(
        self,
        triple_to_explain: Tuple[Any, Any, Any],
        perspective: str,
        triples_to_remove: list,
    ):
        """
        Given a "triple to explain" (that is, a triple that the model currently predicts as true,
        and that we want to be predicted as false);
        given the perspective from which to analyze it;
        and given and a list of training triples containing the entity to convert;
        compute the relevance of the triples in removal, that is, an estimate of the effect they would have
        if removed (all together) from the perspective entity to worsen the prediction of the triple to convert.

        :param triple_to_explain: the triple that we would like the model to predict as "true",
                                  in the form of a tuple (head, relation, tail)
        :param perspective: the perspective from which to explain the fact: it can be either "head" or "tail"
        :param triples_to_remove:   the list of triples containing the perspective entity
                                    that we want to analyze the effect of, if added to the perspective entity

        :return:
        """
        start_time = time.time()
        head_id, relation_id, tail_id = triple_to_explain
        original_entity_to_convert = head_id if perspective == "head" else tail_id

        # check how the original model performs on the original triple to convert
        (
            original_target_entity_score,
            original_best_entity_score,
            original_target_entity_rank,
        ) = self.original_results_for(original_triple_to_predict=triple_to_explain)

        # get from the cache a Kelpie Dataset focused on the original id of the entity to explain,
        # (or create it from scratch if it is not in cache)
        kelpie_dataset = self._get_kelpie_dataset_for(
            original_entity_id=original_entity_to_convert
        )

        kelpie_model_class = self.model.kelpie_model_class()

        kelpie_init_tensor_size = (
            self.model.dimension
            if not isinstance(self.model, TuckER)
            else self.model.entity_dimension
        )
        kelpie_init_tensor = torch.rand(1, kelpie_init_tensor_size)

        # run base post-training to obtain a "clone" of the perspective entity and see how it performs in the triple to convert
        base_kelpie_model = kelpie_model_class(
            model=self.model, dataset=kelpie_dataset, init_tensor=kelpie_init_tensor
        )
        (
            base_pt_target_entity_score,
            base_pt_best_entity_score,
            base_pt_target_entity_rank,
        ) = self.base_post_training_results_for(
            kelpie_model=base_kelpie_model,
            kelpie_dataset=kelpie_dataset,
            original_triple_to_predict=triple_to_explain,
        )

        # run actual post-training by adding the passed triples to the perspective entity and see how it performs in the triple to convert
        pt_kelpie_model = kelpie_model_class(
            model=self.model, dataset=kelpie_dataset, init_tensor=kelpie_init_tensor
        )

        # run actual post-training by adding the passed triples to the perspective entity and see how it performs in the triple to convert
        (
            pt_target_entity_score,
            pt_best_entity_score,
            pt_target_entity_rank,
        ) = self.removal_post_training_results_for(
            kelpie_model=pt_kelpie_model,
            kelpie_dataset=kelpie_dataset,
            original_triple_to_predict=triple_to_explain,
            original_triples_to_remove=triples_to_remove,
        )

        # we want to give higher priority to the facts that, when added, make the score worse (= higher).
        rank_worsening = pt_target_entity_rank - base_pt_target_entity_rank

        # if the model is a minimizer the smaller base_pt_target_entity_score is than pt_target_entity_score, the more relevant the removed facts;
        # if the model is a maximizer, the greater base_pt_target_entity_score is than pt_target_entity_score, the more relevant the removed facts
        score_worsening = (
            pt_target_entity_score - base_pt_target_entity_score
            if self.model.is_minimizer()
            else base_pt_target_entity_score - pt_target_entity_score
        )

        # note: the formulation is very different from the addition one
        relevance = float(rank_worsening + self.sigmoid(score_worsening))

        end_time = time.time()
        execution_time = end_time - start_time
        return (
            relevance,
            original_best_entity_score,
            original_target_entity_score,
            original_target_entity_rank,
            base_pt_best_entity_score,
            base_pt_target_entity_score,
            base_pt_target_entity_rank,
            pt_best_entity_score,
            pt_target_entity_score,
            pt_target_entity_rank,
            execution_time,
        )

    # private methods that know how to access cache structures

    def _get_kelpie_dataset_for(self, original_entity_id: int) -> KelpieDataset:
        """
        Return the value of the queried key in O(1).
        Additionally, move the key to the end to show that it was recently used.

        :param original_entity_id:
        :return:
        """

        if original_entity_id not in self._kelpie_dataset_cache:
            kelpie_dataset = KelpieDataset(
                dataset=self.dataset, entity_id=original_entity_id
            )
            self._kelpie_dataset_cache[original_entity_id] = kelpie_dataset
            self._kelpie_dataset_cache.move_to_end(original_entity_id)

            if len(self._kelpie_dataset_cache) > self._kelpie_dataset_cache_size:
                self._kelpie_dataset_cache.popitem(last=False)

        return self._kelpie_dataset_cache[original_entity_id]

    def original_results_for(self, original_triple_to_predict: numpy.array):
        triple = (
            original_triple_to_predict[0],
            original_triple_to_predict[1],
            original_triple_to_predict[2],
        )
        if not triple in self._original_model_results:
            (
                target_entity_score,
                best_entity_score,
                target_entity_rank,
            ) = self.extract_detailed_performances_on_triple(
                self.model, original_triple_to_predict
            )

            self._original_model_results[triple] = (
                target_entity_score,
                best_entity_score,
                target_entity_rank,
            )

        return self._original_model_results[triple]

    def base_post_training_results_for(
        self,
        kelpie_model: KelpieModel,
        kelpie_dataset: KelpieDataset,
        original_triple_to_predict: numpy.array,
    ):
        """

        :param kelpie_model: an UNTRAINED kelpie model that has just been initialized
        :param kelpie_dataset:
        :param original_triple_to_predict:
        :return:
        """
        original_triple_to_predict = (
            original_triple_to_predict[0],
            original_triple_to_predict[1],
            original_triple_to_predict[2],
        )
        kelpie_triple_to_predict = kelpie_dataset.as_kelpie_triple(
            original_triple=original_triple_to_predict
        )

        if not original_triple_to_predict in self._base_pt_model_results:
            original_entity_name = kelpie_dataset.id_to_entity[
                kelpie_dataset.original_entity_id
            ]
            print(
                "\t\tRunning base post-training on entity "
                + original_entity_name
                + " with no additions"
            )
            base_pt_model = self.post_train(
                kelpie_model_to_post_train=kelpie_model,
                kelpie_training_triples=kelpie_dataset.kelpie_training_triples,
            )  # type: KelpieModel

            # then check how the base post-trained model performs on the kelpie triple to explain.
            # This means checking how the "clone entity" (with no additional triples) performs
            (
                base_pt_target_entity_score,
                base_pt_best_entity_score,
                base_pt_target_entity_rank,
            ) = self.extract_detailed_performances_on_triple(
                base_pt_model, kelpie_triple_to_predict
            )

            self._base_pt_model_results[original_triple_to_predict] = (
                base_pt_target_entity_score,
                base_pt_best_entity_score,
                base_pt_target_entity_rank,
            )

        return self._base_pt_model_results[original_triple_to_predict]

    def addition_post_training_results_for(
        self,
        kelpie_model: KelpieModel,
        kelpie_dataset: KelpieDataset,
        original_triple_to_predict: numpy.array,
        original_triples_to_add: numpy.array,
    ):
        """

        :param kelpie_model: an UNTRAINED kelpie model that has just been initialized
        :param kelpie_dataset:
        :param original_triple_to_predict:
        :param original_triples_to_add:
        :return:
        """
        original_triple_to_predict = (
            original_triple_to_predict[0],
            original_triple_to_predict[1],
            original_triple_to_predict[2],
        )
        kelpie_triple_to_predict = kelpie_dataset.as_kelpie_triple(
            original_triple=original_triple_to_predict
        )

        # these are original triples, and not "kelpie" triples.
        # the "add_training_triples" method replaces the original entity with the kelpie entity by itself
        kelpie_dataset.add_training_triples(original_triples_to_add)

        original_entity_name = kelpie_dataset.id_to_entity[
            kelpie_dataset.original_entity_id
        ]
        print(
            "\t\tRunning post-training on entity "
            + original_entity_name
            + " adding triples: "
        )
        for x in original_triples_to_add:
            print("\t\t\t" + kelpie_dataset.printable_triple(x))

        # post-train the kelpie model on the dataset that has undergone the addition
        cur_kelpie_model = self.post_train(
            kelpie_model_to_post_train=kelpie_model,
            kelpie_training_triples=kelpie_dataset.kelpie_training_triples,
        )  # type: KelpieModel

        # then check how the post-trained model performs on the kelpie triple to explain.
        # This means checking how the "kelpie entity" (with the added triple) performs, rather than the original entity
        (
            pt_target_entity_score,
            pt_best_entity_score,
            pt_target_entity_rank,
        ) = self.extract_detailed_performances_on_triple(
            cur_kelpie_model, kelpie_triple_to_predict
        )

        # undo the addition, to allow the following iterations of this loop
        kelpie_dataset.undo_last_training_triples_addition()

        return pt_target_entity_score, pt_best_entity_score, pt_target_entity_rank

    def removal_post_training_results_for(
        self,
        kelpie_model: KelpieModel,
        kelpie_dataset: KelpieDataset,
        original_triple_to_predict: numpy.array,
        original_triples_to_remove: numpy.array,
    ):
        """
        :param kelpie_model: an UNTRAINED kelpie model that has just been initialized
        :param kelpie_dataset:
        :param original_triple_to_predict:
        :param original_triples_to_remove:
        :return:
        """

        original_triple_to_predict = (
            original_triple_to_predict[0],
            original_triple_to_predict[1],
            original_triple_to_predict[2],
        )
        kelpie_triple_to_predict = kelpie_dataset.as_kelpie_triple(
            original_triple=original_triple_to_predict
        )

        # these are original triples, and not "kelpie" triples.
        # the "remove_training_triples" method replaces the original entity with the kelpie entity by itself
        kelpie_dataset.remove_training_triples(original_triples_to_remove)

        original_entity_name = kelpie_dataset.id_to_entity[
            kelpie_dataset.original_entity_id
        ]
        print(
            "\t\tRunning post-training on entity "
            + original_entity_name
            + " removing triples: "
        )
        for x in original_triples_to_remove:
            print("\t\t\t" + kelpie_dataset.printable_triple(x))

        # post-train a kelpie model on the dataset that has undergone the removal
        cur_kelpie_model = self.post_train(
            kelpie_model_to_post_train=kelpie_model,
            kelpie_training_triples=kelpie_dataset.kelpie_training_triples,
        )  # type: KelpieModel

        # then check how the post-trained model performs on the kelpie triple to explain.
        # This means checking how the "kelpie entity" (without the removed triples) performs,
        # rather than the original entity
        (
            pt_target_entity_score,
            pt_best_entity_score,
            pt_target_entity_rank,
        ) = self.extract_detailed_performances_on_triple(
            cur_kelpie_model, kelpie_triple_to_predict
        )

        # undo the removal, to allow the following iterations of this loop
        kelpie_dataset.undo_last_training_triples_removal()

        return pt_target_entity_score, pt_best_entity_score, pt_target_entity_rank

    # private methods to do stuff

    def post_train(
        self, kelpie_model_to_post_train: KelpieModel, kelpie_training_triples: numpy.array
    ):
        """

        :param kelpie_model_to_post_train: an UNTRAINED kelpie model that has just been initialized
        :param kelpie_training_triples:
        :return:
        """
        # kelpie_model_class = self.model.kelpie_model_class()
        # kelpie_model = kelpie_model_class(model=self.model, dataset=kelpie_dataset)
        kelpie_model_to_post_train.to("cuda")

        optimizer = self.kelpie_optimizer_class(
            model=kelpie_model_to_post_train,
            hyperparameters=self.hyperparameters,
            verbose=False,
        )
        optimizer.train(training_triples=kelpie_training_triples)
        return kelpie_model_to_post_train

    def extract_detailed_performances_on_triple(
        self, model: Model, triple: numpy.array
    ):
        model.eval()
        head_id, relation_id, tail_id = triple

        # check how the model performs on the triple to explain
        all_scores = model.all_scores(numpy.array([triple])).detach().cpu().numpy()[0]
        target_entity_score = all_scores[
            tail_id
        ]  # todo: this only works in "head" perspective
        filter_out = (
            model.dataset.to_filter[(head_id, relation_id)]
            if (head_id, relation_id) in model.dataset.to_filter
            else []
        )

        if model.is_minimizer():
            all_scores[filter_out] = 1e6
            # if the target score had been filtered out, put it back
            # (this may happen in necessary mode, where we may run this method on the actual test triple;
            # not in sufficient mode, where we run this method on the unseen "triples to convert")
            all_scores[tail_id] = target_entity_score
            best_entity_score = numpy.min(all_scores)
            target_entity_rank = numpy.sum(
                all_scores <= target_entity_score
            )  # we use min policy here

        else:
            all_scores[filter_out] = -1e6
            # if the target score had been filtered out, put it back
            # (this may happen in necessary mode, where we may run this method on the actual test triple;
            # not in sufficient mode, where we run this method on the unseen "triples to convert")
            all_scores[tail_id] = target_entity_score
            best_entity_score = numpy.max(all_scores)
            target_entity_rank = numpy.sum(
                all_scores >= target_entity_score
            )  # we use min policy here

        return target_entity_score, best_entity_score, target_entity_rank
