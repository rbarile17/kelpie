import numpy

from typing import Tuple, Any

from .engine import ExplanationEngine

from ..data import Dataset
from ..link_prediction.models import Model, TuckER


class DataPoisoningEngine(ExplanationEngine):
    def __init__(
        self, model: Model, dataset: Dataset, hyperparameters: dict, epsilon: float
    ):
        ExplanationEngine.__init__(
            self, model=model, dataset=dataset, hyperparameters=hyperparameters
        )
        self.epsilon = epsilon
        self.lambd = 1

        # cache (triple, entity) -> gradient
        self.gradients_cache = {}

    def removal_relevance(
        self,
        triple_to_explain: Tuple[Any, Any, Any],
        perspective: str,
        triples_to_remove: list,
    ):
        if len(triples_to_remove) > 1:
            raise NotImplementedError(
                "Data Poisoning Engine only supports single triple removal"
            )

        triple_to_remove = triples_to_remove[0]

        head_id, relation_id, tail_id = triple_to_explain
        entity_to_explain_id = head_id if perspective == "head" else tail_id

        # this is EXTREMELY important in models with dropout and/or batch normalization.
        # It disables dropout, and tells batch_norm layers to use saved statistics instead of batch data.
        # This affects both the computed scores and the computed gradients.
        self.model.eval()

        # get score and rank of the triple to explain
        (
            (triple_to_explain_direct_score, _),
            (_, triple_to_explain_target_rank),
            _,
        ) = self.model.predict_triple(numpy.array(triple_to_explain))

        # compute the gradient of the fact to explain with respect to the embedding of the entity to explain.
        # the gradient points towards the direction that INCREASES the score.
        gradient = self.get_gradient_for(
            triple=triple_to_explain, entity_to_explain=entity_to_explain_id
        )

        # move the embedding of the entity to explain in the direction that worsens the score
        # (that is, the direction that increases it if the model minimizes the scores of true facts,
        # or the direction that decreases if the model maximizes the scores of true facts)
        if self.model.is_minimizer():
            perturbed_entity_to_explain_embedding = (
                self.model.entity_embeddings[entity_to_explain_id].detach()
                + self.epsilon * gradient.detach()
            )
        else:
            perturbed_entity_to_explain_embedding = (
                self.model.entity_embeddings[entity_to_explain_id].detach()
                - self.epsilon * gradient.detach()
            )

        # create a numpy array that just features the triple to remove twice
        triples_numpy_array = numpy.array([triple_to_remove, triple_to_remove])

        # get the original and perturbed embeddings for head, relation and tail of the triple to remove
        head_embeddings_tensor = self.model.entity_embeddings[triples_numpy_array[:, 0]]
        rel_embeddings_tensor = self.model.relation_embeddings[
            triples_numpy_array[:, 1]
        ]
        tail_embeddings_tensor = self.model.entity_embeddings[triples_numpy_array[:, 2]]
        if triple_to_remove[0] == entity_to_explain_id:
            head_embeddings_tensor[1] = perturbed_entity_to_explain_embedding
        else:
            tail_embeddings_tensor[1] = perturbed_entity_to_explain_embedding

        # compute the original score (using the original embeddings)
        # and the perturbed score (using the perturbed embeddings)
        scores = (
            self.model.score_embeddings(
                head_embeddings_tensor, rel_embeddings_tensor, tail_embeddings_tensor
            )
            .detach()
            .cpu()
            .numpy()
        )
        original_triple_to_remove_score, perturbed_triple_to_remove_score = (
            scores[0],
            scores[1],
        )

        # Relevance = original score - lambda * perturbed score
        # (that is: we have tweaked a little bit the embedding of the perspective entity in order to worsen the fact to explain:
        # the more a training fact gets worsened as well, the more relevant it was.)
        if self.model.is_minimizer():
            relevance = (
                -original_triple_to_remove_score
                + self.lambd * perturbed_triple_to_remove_score
            )
        else:
            relevance = (
                original_triple_to_remove_score
                - self.lambd * perturbed_triple_to_remove_score
            )

        return (
            relevance,
            triple_to_explain_direct_score,
            triple_to_explain_target_rank,
            original_triple_to_remove_score,
            perturbed_triple_to_remove_score,
        )

    def addition_relevance(
        self,
        triple_to_convert: Tuple[Any, Any, Any],
        perspective: str,
        triples_to_add: list,
    ):
        if len(triples_to_add) > 1:
            raise NotImplementedError(
                "Data Poisoning Engine only supports single triple removal"
            )

        triple_to_add = triples_to_add[0]

        # identify the entity to explain
        head_id, relation_id, tail_id = triple_to_convert
        entity_to_explain_id = head_id if perspective == "head" else tail_id

        (
            (triple_to_convert_direct_score, _),
            (_, triple_to_convert_target_rank),
            _,
        ) = self.model.predict_triple(numpy.array(triple_to_convert))

        # this is EXTREMELY important in models with dropout and/or batch normalization.
        # It disables dropout, and tells batch_norm layers to use saved statistics instead of batch data.
        # This affects both the computed scores and the computed gradients.
        self.model.eval()

        # compute the gradient of the fact to convert with respect to the embedding of the entity to explain
        gradient = self.get_gradient_for(
            triple=triple_to_convert, entity_to_explain=entity_to_explain_id
        )

        # Shift the embedding of the entity to explain in the direction that improves the plausibility of the fact to convert.
        # If the model is a minimizer, this means a shift in opposite direction to the score gradient
        if self.model.is_minimizer():
            perturbed_entity_to_explain_embedding = (
                self.model.entity_embeddings[entity_to_explain_id].detach()
                - self.epsilon * gradient.detach()
            )
        # If the model is a maximizer, this means going in the same direction as the score gradient
        else:
            perturbed_entity_to_explain_embedding = (
                self.model.entity_embeddings[entity_to_explain_id].detach()
                + self.epsilon * gradient.detach()
            )

        # create a numpy array that just features the triple to add twice
        triples_numpy_array = numpy.array([triple_to_add, triple_to_add])

        # get the original and perturbed embeddings for head, relation and tail of the triple to add
        head_embeddings_tensor = self.model.entity_embeddings[triples_numpy_array[:, 0]]
        rel_embeddings_tensor = self.model.relation_embeddings[
            triples_numpy_array[:, 1]
        ]
        tail_embeddings_tensor = self.model.entity_embeddings[triples_numpy_array[:, 2]]
        if triple_to_add[0] == entity_to_explain_id:
            head_embeddings_tensor[1] = perturbed_entity_to_explain_embedding
        else:
            tail_embeddings_tensor[1] = perturbed_entity_to_explain_embedding

        # compute the original score (using the original embeddings)
        # and the perturbed score (using the perturbed embeddings)
        scores = (
            self.model.score_embeddings(
                head_embeddings_tensor, rel_embeddings_tensor, tail_embeddings_tensor
            )
            .detach()
            .cpu()
            .numpy()
        )
        original_triple_to_add_score, perturbed_triple_to_add_score = (
            scores[0],
            scores[1],
        )

        # Relevance = original score - lambda * perturbed score
        # (that is: we have tweaked a little bit the embedding of the perspective entity in order to improve the fact to cpnvert:
        # the more a fact to add gets improved as well, the more relevant it will be in addition.)
        if self.model.is_minimizer():
            relevance = (
                original_triple_to_add_score
                - self.lambd * perturbed_triple_to_add_score
            )
        else:
            relevance = (
                -original_triple_to_add_score
                + self.lambd * perturbed_triple_to_add_score
            )

        return (
            relevance,
            triple_to_convert_direct_score,
            triple_to_convert_target_rank,
            original_triple_to_add_score,
            perturbed_triple_to_add_score,
        )

    def get_gradient_for(self, triple: Tuple[Any, Any, Any], entity_to_explain: int):
        triple_head, triple_relation, triple_tail = triple
        assert entity_to_explain in ([triple_head, triple_tail])

        if (
            (triple_head, triple_relation, triple_tail),
            entity_to_explain,
        ) in self.gradients_cache:
            return self.gradients_cache[
                ((triple_head, triple_relation, triple_tail), entity_to_explain)
            ].cuda()

        entity_dimension = (
            self.model.entity_dimension
            if isinstance(self.model, TuckER)
            else self.model.dimension
        )
        relation_dimension = (
            self.model.relation_dimension
            if isinstance(self.model, TuckER)
            else self.model.dimension
        )

        triple_head_embedding = (
            self.model.entity_embeddings[triple_head]
            .detach()
            .reshape(1, entity_dimension)
        )
        triple_relation_embedding = (
            self.model.relation_embeddings[triple_relation]
            .detach()
            .reshape(1, relation_dimension)
        )
        triple_tail_embedding = (
            self.model.entity_embeddings[triple_tail]
            .detach()
            .reshape(1, entity_dimension)
        )

        cur_entity_to_explain_embedding = (
            triple_head_embedding
            if entity_to_explain == triple_head
            else triple_tail_embedding
        )
        cur_entity_to_explain_embedding.requires_grad = True

        current_score = self.model.score_embeddings(
            triple_head_embedding, triple_relation_embedding, triple_tail_embedding
        )
        current_score.backward()
        current_gradient = cur_entity_to_explain_embedding.grad[0]
        cur_entity_to_explain_embedding.grad = (
            None  # reset the gradient, just to be sure
        )

        # update the cache
        self.gradients_cache[
            ((triple_head, triple_relation, triple_tail), entity_to_explain)
        ] = current_gradient.cpu()
        return current_gradient.cuda()
