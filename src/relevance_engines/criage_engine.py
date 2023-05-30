import random
import numpy
import torch

from collections import defaultdict
from typing import Tuple, Any

from .engine import ExplanationEngine

from ..data import Dataset, ONE_TO_ONE, MANY_TO_ONE
from ..link_prediction.models import Model, ComplEx, ConvE, DistMult


class CriageEngine(ExplanationEngine):
    def __init__(self, model: Model, dataset: Dataset, hyperparameters: dict):
        ExplanationEngine.__init__(
            self, model=model, dataset=dataset, hyperparameters=hyperparameters
        )

        if (
            not isinstance(model, ComplEx)
            and not isinstance(model, ConvE)
            and not isinstance(model, DistMult)
        ):
            raise Exception("Criage does not support this model.")

        self.entity_dimension = self.model.dimension

        self.tail_to_training_triples = defaultdict(list)
        for h, r, t in dataset.training_triples:
            self.tail_to_training_triples[t].append((h, r, t))

        # caches
        self.hr_2_z = {}
        self.entity_2_hessian = {}

    def removal_relevance(
        self,
        triple_to_explain: Tuple[Any, Any, Any],
        perspective: str,
        triples_to_remove: list,
    ):
        if len(triples_to_remove) > 1:
            raise NotImplementedError(
                "Criage Engine only supports single triple removal"
            )

        self.model.eval()

        triple_to_remove = triples_to_remove[0]
        (head_to_explain, relation_to_explain, tail_to_explain) = triple_to_explain

        # this means that the tail of triple_to_remove is the tail of the triple to explain
        if perspective == "tail":
            assert tail_to_explain in self.tail_to_training_triples

            z_head_relation = self.get_z_for(triple_to_explain)

            hessian_matrix = self.get_hessian_for(
                entity=tail_to_explain,
                training_triples=self.tail_to_training_triples.get(
                    tail_to_explain, []
                ),
            )

            z_cur_head_relation = self.get_z_for(triple_to_remove)

            score_variation, _ = self._estimate_score_variation_in_removal(
                z_triple_to_explain=z_head_relation,
                z_triple_to_remove=z_cur_head_relation,
                entity_id=tail_to_explain,
                entity_hessian_matrix=hessian_matrix,
            )

        # this means that the tail of triple_to_remove is the head of the triple to explain
        elif perspective == "head":
            assert head_to_explain in self.tail_to_training_triples

            z_tail_relation = self.get_z_for(
                (tail_to_explain, relation_to_explain, head_to_explain)
            )

            hessian_matrix = self.get_hessian_for(
                entity=head_to_explain,
                training_triples=self.tail_to_training_triples.get(
                    head_to_explain, []
                ),
            )

            z_cur_head_relation = self.get_z_for(triple_to_remove)
            score_variation, _ = self._estimate_score_variation_in_removal(
                z_triple_to_explain=z_tail_relation,
                z_triple_to_remove=z_cur_head_relation,
                entity_id=head_to_explain,
                entity_hessian_matrix=hessian_matrix,
            )
        else:
            raise ValueError

        # we want the score variation has to be as NEGATIVELY large as possible
        return score_variation

    def addition_relevance(
        self,
        triple_to_convert: Tuple[Any, Any, Any],
        perspective: str,
        triples_to_add: list,
    ):
        if len(triples_to_add) > 1:
            raise NotImplementedError(
                "Criage Engine only supports single triple addition"
            )

        self.model.eval()

        triple_to_add = triples_to_add[0]
        (head_to_convert, relation_to_convert, tail_to_convert) = triple_to_convert

        # this means that the tail of triple_to_add is the tail_to_convert
        if perspective == "tail":
            assert triple_to_add[2] == triple_to_convert[2]

            z_head_relation = self.get_z_for(triple_to_convert)

            training_triples_with_tail_as_tail = []
            if tail_to_convert in self.tail_to_training_triples:
                training_triples_with_tail_as_tail = (
                    self.tail_to_training_triples[tail_to_convert]
                )

            hessian_matrix = self.get_hessian_for(
                entity=tail_to_convert,
                training_triples=training_triples_with_tail_as_tail,
            )

            z_cur_head_relation = self.get_z_for(triple_to_add)
            # z_cur_head_relation = self.get_z_for(triple_to_remove).detach().cpu().numpy()

            score_variation, _ = self._estimate_score_variation_in_addition(
                z_triple_to_explain=z_head_relation,
                z_triple_to_add=z_cur_head_relation,
                entity_id=tail_to_convert,
                entity_hessian_matrix=hessian_matrix,
            )

        # this means that the tail of triple_to_add is the head_to_convert
        elif perspective == "head":
            assert triple_to_add[2] == triple_to_convert[0]

            z_tail_relation = self.get_z_for(
                (tail_to_convert, relation_to_convert, head_to_convert)
            )

            training_triples_with_head_as_tail = []
            if head_to_convert in self.tail_to_training_triples:
                training_triples_with_head_as_tail = (
                    self.tail_to_training_triples[head_to_convert]
                )

            hessian_matrix = self.get_hessian_for(
                entity=head_to_convert,
                training_triples=training_triples_with_head_as_tail,
            )

            z_cur_head_relation = self.get_z_for(triple_to_add)
            score_variation, _ = self._estimate_score_variation_in_addition(
                z_triple_to_explain=z_tail_relation,
                z_triple_to_add=z_cur_head_relation,
                entity_id=head_to_convert,
                entity_hessian_matrix=hessian_matrix,
            )
        else:
            raise ValueError

        # we want the score variation has to be as large as possible
        return score_variation

    def get_z_for(self, triple):
        head, relation, _ = triple
        hr = (head, relation)
        if hr not in self.hr_2_z:
            self.hr_2_z[hr] = self.model.criage_first_step(numpy.array([triple]))
        return self.hr_2_z[hr]

    def get_hessian_for(self, entity, training_triples):
        if entity not in self.entity_2_hessian:
            self.entity_2_hessian[entity] = self.compute_hessian_for(
                entity, training_triples
            )
        return self.entity_2_hessian[entity]

    def sigmoid(self, x):
        return 1 / (1 + numpy.exp(-x))

    def compute_hessian_for(self, entity, triples_featuring_entity_as_tail):
        """
        This method computes the Hessian matrix for an entity, based on the triples that feature that entity as tail

        :param entity: the id of the entity to compute the hessian for
        :param triples_featuring_entity_as_tail: the list of triples featuring the passed entity as tail
        :return: the computed Hessian matrix, that is, an entity_dimension x entity_dimension matrix
        """

        all_entity_embeddings = self.model.entity_embeddings
        all_relation_embeddings = self.model.relation_embeddings

        entity_embedding = all_entity_embeddings[entity].detach().cpu().numpy()

        # initialize the Hessian matrix with zeros
        hessian_matrix = numpy.zeros((self.entity_dimension, self.entity_dimension))

        for triple in triples_featuring_entity_as_tail:
            (head_id, relation_id, _) = triple

            head_embedding = all_entity_embeddings[head_id].detach().cpu().numpy()
            relation_embedding = (
                all_relation_embeddings[relation_id].detach().cpu().numpy()
            )

            x = numpy.multiply(
                numpy.reshape(head_embedding, (1, -1)),
                numpy.reshape(relation_embedding, (1, -1)),
            )

            x_2 = numpy.dot(entity_embedding, numpy.transpose(x))

            sig_tri = self.sigmoid(x_2)
            sig = sig_tri * (1 - sig_tri)
            hessian_matrix += sig * numpy.dot(numpy.transpose(x), x)
        return hessian_matrix

    def _estimate_score_variation_in_addition(
        self, z_triple_to_explain, z_triple_to_add, entity_id, entity_hessian_matrix
    ):
        entity_embedding = (
            self.model.entity_embeddings[entity_id].detach().cpu().numpy()
        )

        z_triple_to_add = z_triple_to_add.detach().cpu().numpy()
        z_triple_to_explain = z_triple_to_explain.detach().cpu().numpy()
        x_2 = numpy.dot(entity_embedding, numpy.transpose(z_triple_to_add))
        sig_tri = self.sigmoid(x_2)

        m = numpy.linalg.inv(
            entity_hessian_matrix
            + sig_tri
            * (1 - sig_tri)
            * numpy.dot(numpy.transpose(z_triple_to_add), z_triple_to_add)
        )
        relevance = numpy.dot(
            z_triple_to_explain,
            numpy.transpose((1 - sig_tri) * numpy.dot(z_triple_to_add, m)),
        )

        return relevance[0][0], m

    def _estimate_score_variation_in_removal(
        self, z_triple_to_explain, z_triple_to_remove, entity_id, entity_hessian_matrix
    ):
        entity_embedding = (
            self.model.entity_embeddings[entity_id].detach().cpu().numpy()
        )

        z_triple_to_remove = z_triple_to_remove.detach().cpu().numpy()
        z_triple_to_explain = z_triple_to_explain.detach().cpu().numpy()

        x_2 = numpy.dot(entity_embedding, numpy.transpose(z_triple_to_remove))
        sig_tri = self.sigmoid(x_2)

        try:
            m = numpy.linalg.inv(
                entity_hessian_matrix
                + sig_tri
                * (1 - sig_tri)
                * numpy.dot(numpy.transpose(z_triple_to_remove), z_triple_to_remove)
            )
            relevance = numpy.dot(
                z_triple_to_explain,
                numpy.transpose((1 - sig_tri) * numpy.dot(z_triple_to_remove, m)),
            )

            return -relevance[0][0], m

        except Exception:
            print(self.dataset.entity_id_2_name[entity_id])

    def extract_entities_for(
        self,
        model: Model,
        dataset: Dataset,
        triple: numpy.array,
        perspective: str,
        k: int,
        degree_cap=-1,
    ):
        # this is EXTREMELY important in models with dropout and/or batch normalization.
        # It basically disables dropout, and tells batch_norm layers to use saved statistics instead of batch data.
        # (This affects both the computed scores and the computed gradients, so it is vital)
        model.eval()

        # disable backprop for all the following operations: hopefully this should make them faster
        with torch.no_grad():
            head_to_explain, relation_to_explain, tail_to_explain = triple
            entity_to_explain, _ = (
                (relation_to_explain, tail_to_explain)
                if perspective == "head"
                else (tail_to_explain, head_to_explain)
            )

            overall_candidate_entities = []

            if perspective == "head":
                step_1_candidate_entities = []
                step_1_triples = []
                for cur_entity in range(0, dataset.num_entities):
                    # do not include the entity to explain, of course
                    if cur_entity == entity_to_explain:
                        continue

                    # if the entity only appears in validation/testing (so its training degree is 0) skip it
                    if dataset.entity_to_degree[cur_entity] < 1:
                        continue

                    # if the training degree exceeds the cap, skip the entity
                    if (
                        degree_cap != -1
                        and dataset.entity_to_degree[cur_entity] > degree_cap
                    ):
                        continue

                    # THIS IS IMPORTANT IN CRIAGE
                    # if the entity does not appear as tail in training even once, ignore it
                    if cur_entity not in self.tail_to_training_triples:
                        continue

                    # if any facts <cur_entity, relation, *> are in the dataset:
                    if (cur_entity, relation_to_explain) in dataset.to_filter:
                        # if the relation is *_TO_ONE, ignore any entities for which in train/valid/test,
                        if dataset.relation_to_type[relation_to_explain] in [
                            ONE_TO_ONE,
                            MANY_TO_ONE,
                        ]:
                            continue

                        # if <cur_entity, relation, tail> is in the dataset, ignore this entity
                        if (
                            tail_to_explain
                            in dataset.to_filter[(cur_entity, relation_to_explain)]
                        ):
                            continue

                    step_1_candidate_entities.append(cur_entity)
                    step_1_triples.append(
                        (cur_entity, relation_to_explain, tail_to_explain)
                    )

                if len(step_1_candidate_entities) == 0:
                    return []

                batch_size = 500
                # if isinstance(model, TransE) and len(step_1_triples) > batch_size:
                batch_scores_array = []
                batch_start = 0
                while batch_start < len(step_1_triples):
                    cur_batch = step_1_triples[
                        batch_start : min(len(step_1_triples), batch_start + batch_size)
                    ]
                    cur_batch_all_scores = (
                        model.all_scores(triples=numpy.array(cur_batch))
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    batch_scores_array.append(cur_batch_all_scores)
                    batch_start += batch_size
                triples_all_scores = numpy.vstack(batch_scores_array)

                for i in range(len(step_1_candidate_entities)):
                    cur_candidate_entity = step_1_candidate_entities[i]
                    cur_head, cur_rel, cur_tail = step_1_triples[i]
                    cur_triple_all_scores = triples_all_scores[i]

                    filter_out = (
                        dataset.to_filter[(cur_head, cur_rel)]
                        if (cur_head, cur_rel) in dataset.to_filter
                        else []
                    )

                    if model.is_minimizer():
                        cur_triple_all_scores[torch.LongTensor(filter_out)] = 1e6
                        cur_triple_target_score_filtered = cur_triple_all_scores[
                            cur_tail
                        ]
                        if (
                            1e6
                            > cur_triple_target_score_filtered
                            > numpy.min(cur_triple_all_scores)
                        ):
                            overall_candidate_entities.append(cur_candidate_entity)
                    else:
                        cur_triple_all_scores[torch.LongTensor(filter_out)] = -1e6
                        cur_triple_target_score_filtered = cur_triple_all_scores[
                            cur_tail
                        ]
                        if (
                            -1e6
                            < cur_triple_target_score_filtered
                            < numpy.max(cur_triple_all_scores)
                        ):
                            overall_candidate_entities.append(cur_candidate_entity)

            else:
                # todo: this is currently not allowed because we would need to collect (head, relation, entity) for all other entities
                # todo: add an optional boolean "head_prediction" (default=False); if it is true, compute scores for all heads rather than tails
                raise NotImplementedError

        return random.sample(
            overall_candidate_entities, k=min(k, len(overall_candidate_entities))
        )