import math
import time
import numpy as np
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
        """PostTrainingEngine constructor.

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
            raise Exception("Already a post-trainable KelpieModel.")

        self.base_pt_model_results = {}
        self.kelpie_dataset_cache_size = 20
        self.kelpie_dataset_cache = OrderedDict()

    def addition_relevance(
        self,
        triple_to_convert: Tuple[Any, Any, Any],
        perspective: str,
        triples_to_add: list,
    ):
        """Given a "triple to convert" (that is, a triple that the model currently does not predict as true,
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

        head, _, tail = triple_to_convert
        original_entity = head if perspective == "head" else tail

        kelpie_dataset = self._get_kelpie_dataset(original_entity=original_entity)

        kelpie_model_class = self.model.kelpie_model_class()
        init_tensor = torch.rand(1, self.model.dimension)

        # run base post-training on homologous mimic of the perspective entity
        # and check how the model performs on the triple to explain
        kelpie_model = kelpie_model_class(
            model=self.model, dataset=kelpie_dataset, init_tensor=init_tensor
        )
        base_pt_results = self.base_post_training_results_for(
            model=kelpie_model,
            dataset=kelpie_dataset,
            triple_to_predict=triple_to_convert,
        )

        # run actual post-training on non homologous mimic of the perspective entity
        # and see how the model performs on the triple to explain
        pt_model = kelpie_model_class(
            model=self.model, dataset=kelpie_dataset, init_tensor=init_tensor
        )
        pt_results = self.addition_post_training_results_for(
            model=pt_model,
            dataset=kelpie_dataset,
            triple_to_predict=triple_to_convert,
            triples_to_add=triples_to_add,
        )

        rank_improvement = base_pt_results["target_rank"] - pt_results["target_rank"]
        score_improvement = (
            base_pt_results["target_score"] - pt_results["target_score"]
            if self.model.is_minimizer()
            else pt_results["target_score"] - base_pt_results["target_score"]
        )

        relevance = float(rank_improvement + self.sigmoid(score_improvement))
        relevance /= float(base_pt_results["target_rank"])

        print(f"\t\tObtained individual relevance: {relevance:.3f}")

        return (
            relevance,
            time.time() - start_time,
        )

    def removal_relevance(
        self,
        triple_to_explain: Tuple[Any, Any, Any],
        perspective: str,
        triples_to_remove: list,
    ):
        """Given a "triple to explain" (that is, a triple that the model currently predicts as true,
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
        head, _, tail = triple_to_explain
        entity_to_convert = head if perspective == "head" else tail

        kelpie_dataset = self._get_kelpie_dataset(original_entity=entity_to_convert)

        kelpie_model_class = self.model.kelpie_model_class()
        init_tensor = torch.rand(1, self.model.dimension)

        # run base post-training on homologous mimic of the perspective entity
        # and check how the model performs on the triple to explain
        base_model = kelpie_model_class(
            model=self.model,
            dataset=kelpie_dataset,
            init_tensor=init_tensor,
        )
        base_pt_results = self.base_post_training_results_for(
            model=base_model,
            dataset=kelpie_dataset,
            triple_to_predict=triple_to_explain,
        )

        # run actual post-training by adding the passed triples to the perspective entity and see how it performs in the triple to convert
        pt_kelpie_model = kelpie_model_class(
            model=self.model, dataset=kelpie_dataset, init_tensor=init_tensor
        )
        pt_results = self.removal_post_training_results_for(
            model=pt_kelpie_model,
            dataset=kelpie_dataset,
            triple_to_predict=triple_to_explain,
            triples_to_remove=triples_to_remove,
        )

        rank_worsening = pt_results["target_rank"] - base_pt_results["target_rank"]
        score_worsening = (
            pt_results["target_score"] - base_pt_results["target_score"]
            if self.model.is_minimizer()
            else base_pt_results["target_score"] - pt_results["target_score"]
        )

        relevance = float(rank_worsening + self.sigmoid(score_worsening))

        return (
            relevance,
            time.time() - start_time,
        )

    # private methods that know how to access cache structures
    def _get_kelpie_dataset(self, original_entity: int) -> KelpieDataset:
        """Return the value of the queried key in O(1).
        Additionally, move the key to the end to show that it was recently used.

        :param original_entity_id:
        :return:
        """

        if original_entity not in self.kelpie_dataset_cache:
            kelpie_dataset = KelpieDataset(
                dataset=self.dataset, entity_id=original_entity
            )
            self.kelpie_dataset_cache[original_entity] = kelpie_dataset
            self.kelpie_dataset_cache.move_to_end(original_entity)

            if len(self.kelpie_dataset_cache) > self.kelpie_dataset_cache_size:
                self.kelpie_dataset_cache.popitem(last=False)

        return self.kelpie_dataset_cache[original_entity]

    def base_post_training_results_for(
        self,
        model: KelpieModel,
        dataset: KelpieDataset,
        triple_to_predict,
    ):
        """Run base post-training on the given model, and return the results on the given triple.
        :param kelpie_model: an UNTRAINED kelpie model that has just been initialized
        :param kelpie_dataset:
        :param original_triple_to_predict:
        :return:
        """

        kelpie_triple = dataset.as_kelpie_triple(triple_to_predict)

        if not triple_to_predict in self.base_pt_model_results:
            entity_name = dataset.id_to_entity[dataset.original_entity_id]
            print(
                f"\tRunning base post-training on entity {entity_name} with no changes"
            )
            base_pt_model = self.post_train(
                model=model,
                training_triples=dataset.kelpie_training_triples,
            )

            # Checking how the model performs on the homologous mimic
            results = self.triple_results(base_pt_model, kelpie_triple)
            self.base_pt_model_results[triple_to_predict] = results

        return self.base_pt_model_results[triple_to_predict]

    def addition_post_training_results_for(
        self,
        model: KelpieModel,
        dataset: KelpieDataset,
        triple_to_predict,
        triples_to_add,
    ):
        """

        :param kelpie_model: an UNTRAINED kelpie model that has just been initialized
        :param kelpie_dataset:
        :param original_triple_to_predict:
        :param original_triples_to_add:
        :return:
        """
        kelpie_triple_to_predict = dataset.as_kelpie_triple(triple_to_predict)
        dataset.add_training_triples(triples_to_add)

        original_entity_name = dataset.id_to_entity[dataset.original_entity_id]
        print(
            f"\tRunning post-training on entity {original_entity_name} adding triples: "
        )
        for x in triples_to_add:
            print(f"\t\t {dataset.printable_triple(x)}")

        model = self.post_train(
            model=model,
            training_triples=dataset.kelpie_training_triples,
        )

        results = self.triple_results(model, kelpie_triple_to_predict)

        # undo the addition, to allow the following iterations of this loop
        dataset.undo_last_training_triples_addition()

        return results

    def removal_post_training_results_for(
        self,
        model: KelpieModel,
        dataset: KelpieDataset,
        triple_to_predict: np.array,
        triples_to_remove: np.array,
    ):
        """
        :param kelpie_model: an UNTRAINED kelpie model that has just been initialized
        :param kelpie_dataset:
        :param original_triple_to_predict:
        :param original_triples_to_remove:
        :return:
        """
        kelpie_triple_to_predict = dataset.as_kelpie_triple(triple_to_predict)
        dataset.remove_training_triples(triples_to_remove)

        entity_name = dataset.id_to_entity[dataset.original_entity_id]
        print(f"\tRunning post-training on entity {entity_name} removing triples: ")
        for x in triples_to_remove:
            print(f"\t\t {dataset.printable_triple(x)}")

        model = self.post_train(
            model=model,
            training_triples=dataset.kelpie_training_triples,
        )

        results = self.triple_results(model, kelpie_triple_to_predict)

        # undo the removal, to allow the following iterations of this loop
        dataset.undo_last_training_triples_removal()

        return results

    def post_train(
        self,
        model: KelpieModel,
        training_triples: np.array,
    ):
        """

        :param kelpie_model_to_post_train: an UNTRAINED kelpie model that has just been initialized
        :param kelpie_training_triples:
        :return:
        """
        model.to("cuda")

        optimizer = self.kelpie_optimizer_class(
            model=model,
            hyperparameters=self.hyperparameters,
            verbose=False,
        )
        optimizer.train(training_triples=np.array(training_triples))
        return model

    def triple_results(self, model: Model, triple):
        model.eval()
        head, relation, tail = triple

        all_scores = model.all_scores(np.array([triple])).detach().cpu().numpy()[0]        
        # todo: this only works in "head" perspective
        target_score = all_scores[tail]
        filter_out = model.dataset.to_filter.get((head, relation), [])
        if model.is_minimizer():
            all_scores[filter_out] = 1e6
            all_scores[tail] = target_score
            best_score = np.min(all_scores)
            target_rank = np.sum(all_scores <= target_score)
        else:
            all_scores[filter_out] = -1e6
            best_score = np.max(all_scores)
            target_rank = np.sum(all_scores >= target_score)
            all_scores[tail] = target_score

        return {
            "target_score": target_score,
            "best_score": best_score,
            "target_rank": target_rank,
        }
