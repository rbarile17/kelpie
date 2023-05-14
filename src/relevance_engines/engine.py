import itertools
import random
import numpy
import torch

from typing import Tuple, Any

from ..data import Dataset, ONE_TO_ONE, MANY_TO_ONE
from ..link_prediction.models import Model, TransE


class ExplanationEngine:
    def __init__(self, model: Model, dataset: Dataset, hyperparameters: dict):
        self.model = model
        self.model.to("cuda")
        self.model.eval()
        self.dataset = dataset
        self.hyperparameters = hyperparameters

    def simple_removal_explanations(
        self, triple_to_explain: Tuple[Any, Any, Any], perspective: str, top_k: int
    ):
        pass

    def simple_addition_explanations(
        self,
        triple_to_explain: Tuple[Any, Any, Any],
        perspective: str,
        triples_to_add: list,
    ):
        pass

    def _extract_triple_nples(self, triples: list, n: int):
        return list(itertools.combinations(triples, n))

    def extract_entities_for(
        self,
        model: Model,
        dataset: Dataset,
        triple: numpy.array,
        perspective: str,
        k: int,
        degree_cap=-1,
    ):
        """
        Extract k entities to replace the perspective entity in the passed triple.

        The purpose of such entities is to allow the engine to identify sufficient rules to explain the triple.
        To do so, the engine replaces the perspective entity in the triple with the extracted entities,
        and the engine analyzes the effect of adding/removing fact featuring those entities.

        The whole system works assuming that the extracted entities, when put in the passed triple,
        result in *wrong* facts, that are not predicted as true by the model;
        the purpose of the engine is identify the minimal combination of facts to added to those entities
        in order to "fool" the model and make it predict those "wrong" facts as true.

        As a consequence the extracted entities will adhere to the following criteria:
            - must be different from the perspective entity of the triple to explain (obviously)
            - it must be seen in training (obviously)
            - the extracted entity must form a "true" fact when put in the triple.
              E.g., Michelle can not be a good replacement for Barack in <Barack, parent, Natasha>
              if <Michelle, parent, Natasha> is already present in the dataset (in train, valid or test set)
            - if the relation in the triple has *_TO_ONE type, the extracted entities must not already have
              a known tail for the relation under analysis in the training set.
              (e.g when explaining <Barack, nationality, USA> we can not use entity "Vladimir" to replace Barack
              if <Vladimir, nationality, Russia> is either in train, valid or test set).
            - the extracted entity must not form a "true" fact when put in the triple.
              E.g., Michelle can not be a good replacement for Barack in <Barack, parent, Natasha>
              if <Michelle, parent, Natasha> is already present in the dataset (in train, valid or test set)
            - the extracted entity must not form a fact that is predicted by the model.
              (we want current_other_entities that, without additions, do not predict the target entity with rank 1)
              (e.g. if we are explaining <Barack, nationality, USA>, George is an acceptable "convertible entity"
              only if <George, nationality, ?> does not already rank USA in 1st position!)

        :param model: the model the prediction of which must be explained
        :param dataset: the dataset on which the passed model has been trained
        :param triple: the triple that the engine is trying to explain
        :param perspective: the perspective from which to explain the passed triple: either "head" or "tail".
        :param k: the number of entities to extract
        :param degree_cap:
        :return:
        """

        # this is EXTREMELY important in models with dropout and/or batch normalization.
        # It basically disables dropout, and tells batch_norm layers to use saved statistics instead of batch data.
        # (This affects both the computed scores and the computed gradients, so it is vital)
        model.eval()

        # disable backprop for all the following operations: hopefully this should make them faster
        with torch.no_grad():
            head_to_explain, relation_to_explain, tail_to_explain = triple
            entity_to_explain, target_entity = (
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

                    # if any facts <cur_entity, relation, *> are in the dataset:
                    if (cur_entity, relation_to_explain) in dataset.to_filter:
                        ## if the relation is *_TO_ONE, ignore any entities for which in train/valid/test,
                        if dataset.relation_to_type[relation_to_explain] in [
                            ONE_TO_ONE,
                            MANY_TO_ONE,
                        ]:
                            continue

                        ## if <cur_entity, relation, tail> is in the dataset, ignore this entity
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

                if isinstance(model, TransE) and dataset.name == "YAGO3-10":
                    batch_size = 100
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

                # else:
                #    triples_all_scores = model.all_scores(triples=numpy.array(step_1_triples)).detach().cpu().numpy()

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
