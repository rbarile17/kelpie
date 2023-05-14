import copy
import torch

import numpy as np

from typing import Tuple, Any

from torch import nn
from torch.nn import Parameter
from torch.nn.init import xavier_normal_

from .model import (
    Model,
    KelpieModel,
    ENTITY_DIMENSION,
    RELATION_DIMENSION,
    INPUT_DROPOUT,
    HIDDEN_DROPOUT_1,
    HIDDEN_DROPOUT_2,
)
from ...data import Dataset, KelpieDataset


class TuckER(Model):
    """
    The TuckER class provides a Model implementation in PyTorch for the TuckER system.
    This implementation adheres to paper "TuckER: Tensor Factorization for Knowledge Graph Completion",
    and is largely inspired by the official implementation provided by the authors
    Ivana Balažević, Carl Allen, and Timothy M. Hospedales at https://github.com/ibalazevic/TuckER.

    In training or evaluation, our TuckER class requires triples to be passed as 2-dimensional np.arrays.
    Each row corresponds to a triple and contains the integer ids of its head, relation and tail.
    Only *direct* triples should be passed to the model.
    """

    def __init__(self, dataset: Dataset, hyperparameters: dict, init_random=True):
        """
        Constructor for TuckER model.

        :param dataset: the Dataset on which to train and evaluate the model
        :param hyperparameters: dict with the model hyperparamters. Must contain at least:
                    ENTITY_DIMENSION: entity embedding dimension
                    RELATION_DIMENSION: relation embedding dimension
                    INPUT_DROPOUT: input layer dropout rate
                    HIDDEN_DROPOUT_1: dropout rate after the first hidden layer
                    HIDDEN_DROPOUT_2: dropout rate after the second hidden layer
        """
        # note: the init_random parameter is important because when initializing a KelpieTuckER,
        #       self.entity_embeddings and self.relation_embeddings must not be initialized as Parameters!

        # initialize this object both as a Model and as a nn.Module
        Model.__init__(self, dataset)
        nn.Module.__init__(self)

        self.name = "TuckER"
        self.dataset = dataset
        self.num_entities = dataset.num_entities  # number of entities in dataset
        self.num_relations = 2 * dataset.num_relations  # number of relations in dataset
        self.entity_dimension = hyperparameters[
            ENTITY_DIMENSION
        ]  # entity embedding dimension
        self.relation_dimension = hyperparameters[
            RELATION_DIMENSION
        ]  # relation embedding dimension
        self.input_dropout_rate = hyperparameters[INPUT_DROPOUT]
        self.hidden_dropout_1_rate = hyperparameters[HIDDEN_DROPOUT_1]
        self.hidden_dropout_2_rate = hyperparameters[HIDDEN_DROPOUT_1]

        self.input_dropout = torch.nn.Dropout(self.input_dropout_rate)
        self.hidden_dropout1 = torch.nn.Dropout(self.hidden_dropout_1_rate)
        self.hidden_dropout2 = torch.nn.Dropout(self.hidden_dropout_2_rate)
        self.batch_norm_1 = torch.nn.BatchNorm1d(self.entity_dimension)
        self.batch_norm_2 = torch.nn.BatchNorm1d(self.entity_dimension)

        # create the embeddings for entities and relations as Parameters.
        # We do not use the torch.Embeddings module here in order to keep the code uniform to the KelpieTuckER model,
        # (on which torch.Embeddings can not be used as they do not allow the post-training mechanism).
        # We have verified that this does not affect performances in any way.
        if init_random:
            self.entity_embeddings = Parameter(
                torch.empty(self.num_entities, self.entity_dimension).cuda(),
                requires_grad=True,
            )
            self.relation_embeddings = Parameter(
                torch.empty(self.num_relations, self.relation_dimension).cuda(),
                requires_grad=True,
            )
            self.core_tensor = Parameter(
                torch.tensor(
                    np.random.uniform(
                        -1,
                        1,
                        (
                            self.relation_dimension,
                            self.entity_dimension,
                            self.entity_dimension,
                        ),
                    ),
                    dtype=torch.float,
                ).cuda(),
                requires_grad=True,
            )

            # initialize only the entity_embeddings and relation embeddings wih xavier method
            xavier_normal_(self.entity_embeddings)
            xavier_normal_(self.relation_embeddings)

    def is_minimizer(self):
        """
        This method specifies whether this model aims at minimizing of maximizing scores .
        :return: True if in this model low scores are better than high scores; False otherwise.
        """
        return False

    def score(self, triples: np.array) -> np.array:
        """
        Compute scores for the passed triples
        :param triples: a 2-dimensional numpy array containing the triples to score, one per row
        :return: a monodimensional numpy array containing the score for each passed triple
        """
        # compute scores for each possible tail
        all_scores = self.all_scores(triples)

        # extract from all_scores the specific scores for the initial triples
        triples_scores = []
        for i, (head_index, relation_index, tail_index) in enumerate(triples):
            triples_scores.append(all_scores[i][tail_index])

        return np.array(triples_scores)

    def score_embeddings(
        self,
        head_embeddings: torch.Tensor,
        rel_embeddings: torch.Tensor,
        tail_embeddings: torch.Tensor,
    ):
        """
        Compute scores for the passed triples of head, relation and tail embeddings.
        :param head_embeddings: a torch.Tensor containing the embeddings representing the head entities
        :param rel_embeddings: a torch.Tensor containing the embeddings representing the relations
        :param tail_embeddings: a torch.Tensor containing the embeddings representing the tail entities

        :return: a numpy array containing the scores computed for the passed triples of embeddings
        """

        # NOTE: this method is extremely important, because apart from being called by the ComplEx score(triples) method
        # it is also used to perform several operations on gradients in our GradientEngine
        # as well as the operations from the paper "Data Poisoning Attack against Knowledge Graph Embedding"
        # that we use as a baseline and as a heuristic for our work.

        # batch normalization and reshape of head embeddings
        head_embeddings = self.batch_norm_1(head_embeddings)
        head_embeddings = self.input_dropout(head_embeddings)
        head_embeddings_reshaped = head_embeddings.view(-1, 1, self.entity_dimension)

        # first multiplication with reshape and dropout
        first_multiplication = torch.mm(
            rel_embeddings, self.core_tensor.view(self.relation_dimension, -1)
        )
        first_multiplication_reshaped = first_multiplication.view(
            -1, self.entity_dimension, self.entity_dimension
        )
        first_multiplication_reshaped = self.hidden_dropout1(
            first_multiplication_reshaped
        )

        # second multiplication with reshape, batch norm and dropout
        second_multiplication = torch.bmm(
            head_embeddings_reshaped, first_multiplication_reshaped
        )
        second_multiplication_reshaped = second_multiplication.view(
            -1, self.entity_dimension
        )
        second_multiplication_reshaped = self.batch_norm_2(
            second_multiplication_reshaped
        )
        second_multiplication_reshaped = self.hidden_dropout2(
            second_multiplication_reshaped
        )

        # third multiplication with sigmoid activation
        result = torch.mm(
            second_multiplication_reshaped, tail_embeddings.transpose(1, 0)
        )
        scores = torch.sigmoid(result)

        output_scores = torch.diagonal(scores)
        return output_scores

    def forward(self, triples: np.array):
        """
        Perform forward propagation on the passed triples.

        In the specific case of TuckER, this method just returns the scores
        that each triple obtains with each possible tail.
        This is because TuckER does not require any external Regularizer,
        so only the scores (for all possible tails) are required in training.
        So, in this specific case, the forward method corresponds to the all_scores method.

        :param triples: a 2-dimensional numpy array containing the triples to run forward propagation on, one per row
        :return: a 2-dimensional numpy array that, for each triple, contains a row with the for each passed triple
        """
        return self.all_scores(triples)

    def all_scores(self, triples: np.array):
        """
        For each of the passed triples, compute scores for all possible entities.
        :param triples: a 2-dimensional numpy array containing the triples to score, one per row
        :return: a 2-dimensional numpy array that, for each triple, contains a row with the score for each possible target tail
        """

        head_indexes, relation_indexes = triples[:, 0], triples[:, 1]
        head_embeddings = self.entity_embeddings[head_indexes]
        relation_embeddings = self.relation_embeddings[relation_indexes]
        tail_embeddings = self.entity_embeddings  # all possible tails

        # batch normalization and reshape of head embeddings

        head_embeddings = self.batch_norm_1(head_embeddings)
        head_embeddings = self.input_dropout(head_embeddings)
        head_embeddings_reshaped = head_embeddings.view(-1, 1, self.entity_dimension)

        # first multiplication with reshape and dropout
        first_multiplication = torch.mm(
            relation_embeddings, self.core_tensor.view(self.relation_dimension, -1)
        )
        first_multiplication_reshaped = first_multiplication.view(
            -1, self.entity_dimension, self.entity_dimension
        )
        first_multiplication_reshaped = self.hidden_dropout1(
            first_multiplication_reshaped
        )

        # second multiplication with reshape, batch norm and dropout
        second_multiplication = torch.bmm(
            head_embeddings_reshaped, first_multiplication_reshaped
        )
        second_multiplication_reshaped = second_multiplication.view(
            -1, self.entity_dimension
        )
        second_multiplication_reshaped = self.batch_norm_2(
            second_multiplication_reshaped
        )
        second_multiplication_reshaped = self.hidden_dropout2(
            second_multiplication_reshaped
        )

        # third multiplication with sigmoid activation
        result = torch.mm(
            second_multiplication_reshaped, tail_embeddings.transpose(1, 0)
        )

        scores = torch.sigmoid(result)

        return scores

    def predict_triples(self, triples: np.array) -> Tuple[Any, Any, Any]:
        """
        This method performs prediction on a collection of triples, and returns the corresponding
        scores, ranks and prediction lists.

        All the passed triples must be DIRECT triples in the original dataset.
        (if the Model supports inverse triples as well,
        it should invert the passed triples while running this method)

        :param triples: the direct triples to predict, in numpy array format
        :return: this method returns three lists:
                    - the list of scores for the passed triples,
                                OR IF THE MODEL SUPPORTS INVERSE FACTS
                        the list of couples <direct triple score, inverse triple score>,
                        where the i-th score refers to the i-th triple in the input triples.

                    - the list of couples (head rank, tail rank)
                        where the i-th couple refers to the i-th triple in the input triples.

                    - the list of couples (head_predictions, tail_predictions)
                        where the i-th couple refers to the i-th triple in the input triples.
                        The head_predictions and tail_predictions for each triple
                        are numpy arrays containing all the predicted heads and tails respectively for that triple.
        """

        scores, ranks, predictions = [], [], []  # output data structures
        direct_triples = triples

        # assert all triples are direct
        assert (triples[:, 1] < self.dataset.num_relations).all()

        # invert triples to perform head predictions
        inverse_triples = self.dataset.invert_triples(direct_triples)

        # obtain scores, ranks and predictions both for direct and inverse triples
        inverse_scores, head_ranks, head_predictions = self.predict_tails(
            inverse_triples
        )
        direct_scores, tail_ranks, tail_predictions = self.predict_tails(direct_triples)

        for i in range(direct_triples.shape[0]):
            # add to the scores list a couple containing the scores of the direct and of the inverse triple
            scores += [(direct_scores[i], inverse_scores[i])]

            # add to the ranks list a couple containing the ranks of the head and of the tail
            ranks.append((int(head_ranks[i]), int(tail_ranks[i])))

            # add to the prediction list a couple containing the lists of predictions
            predictions.append((head_predictions[i], tail_predictions[i]))

        return scores, ranks, predictions

    def predict_tails(self, triples):
        """
        This method receives in input a batch of triples
        and returns the corresponding tail scores, tail ranks and tail predictions
        :param triples: the direct triples to predict, in numpy array format
        :return: this method returns three lists:
                        - the list of tails scores for the passed triples,
                            where the i-th score refers to the i-th triple in the input triples.

                        - the list of tail ranks
                            where the i-th rank refers to the i-th triple in the input triples.

                        - the list of tail predictions
                            where the i-th prediction refers to the i-th triple in the input triples.
                            The tail_predictions for each triple
                            are numpy arrays containing all the predicted tails respectively for that triple.
        """

        scores, ranks, pred_out = [], [], []

        batch_size = 128
        for i in range(0, triples.shape[0], batch_size):
            batch = triples[i : min(i + batch_size, len(triples))]

            all_scores = self.all_scores(batch)

            tail_indexes = torch.tensor(
                batch[:, 2]
            ).cuda()  # tails of all passed triples

            # for each triple to predict
            for triple_number, (head_id, relation_id, tail_id) in enumerate(batch):
                tails_to_filter = self.dataset.to_filter[(head_id, relation_id)]

                # score obtained by the correct tail of the triple
                target_tail_score = all_scores[triple_number, tail_id].item()
                scores.append(target_tail_score)

                # set to 0.0 all the predicted values for all the correct tails for that Head-Rel couple
                all_scores[triple_number, tails_to_filter] = 0.0
                # re-set the predicted value for that tail to the original value
                all_scores[triple_number, tail_id] = target_tail_score

            # this amounts to using ORDINAL policy
            sorted_values, sorted_indexes = torch.sort(
                all_scores, dim=1, descending=True
            )
            sorted_indexes = sorted_indexes.cpu().numpy()

            for row in sorted_indexes:
                pred_out.append(row)

            for row in range(batch.shape[0]):
                # rank of the correct target
                rank = np.where(sorted_indexes[row] == tail_indexes[row].item())[0][0]
                ranks.append(rank + 1)

        return scores, ranks, pred_out

    def kelpie_model_class(self):
        return KelpieTuckER


################


class KelpieTuckER(KelpieModel, TuckER):
    def __init__(self, dataset: KelpieDataset, model: TuckER, init_tensor: None):
        TuckER.__init__(
            self,
            dataset=dataset,
            hyperparameters={
                ENTITY_DIMENSION: model.entity_dimension,
                RELATION_DIMENSION: model.relation_dimension,
                INPUT_DROPOUT: model.input_dropout_rate,
                HIDDEN_DROPOUT_1: model.hidden_dropout_1_rate,
                HIDDEN_DROPOUT_2: model.hidden_dropout_2_rate,
            },
            init_random=False,
        )  # NOTE: this is important! if it is set to True,
        # self.entity_embeddings and self.relation_embeddings will be initialized as Parameters
        # and it will not be possible to overwrite them with mere Tensors
        # such as the one resulting from torch.cat(...) and as frozen_relation_embeddings

        self.model = model
        self.original_entity_id = dataset.original_entity_id
        self.kelpie_entity_id = dataset.kelpie_entity_id

        # extract the values of the trained embeddings for entities, relations and core, and freeze them.
        frozen_entity_embeddings = model.entity_embeddings.clone().detach()
        frozen_relation_embeddings = model.relation_embeddings.clone().detach()
        frozen_core = model.core_tensor.clone().detach()

        # the tensor from which to initialize the kelpie_entity_embedding;
        # if it is None it is initialized randomly
        if init_tensor is None:
            init_tensor = torch.rand(1, self.entity_dimension)

        # It is *extremely* important that kelpie_entity_embedding is both a Parameter and an instance variable
        # because the whole approach of the project is to obtain the parameters model params with parameters() method
        # and to pass them to the Optimizer for optimization.
        #
        # If I used .cuda() outside the Parameter, like
        #       self.kelpie_entity_embedding = Parameter(torch.rand(1, 2*self.dimension), requires_grad=True).cuda()
        # IT WOULD NOT WORK because cuda() returns a Tensor, not a Parameter.

        # Therefore kelpie_entity_embedding would not be a Parameter anymore.

        self.kelpie_entity_embedding = Parameter(init_tensor.cuda(), requires_grad=True)
        self.entity_embeddings = torch.cat(
            [frozen_entity_embeddings, self.kelpie_entity_embedding], 0
        )
        self.relation_embeddings = frozen_relation_embeddings
        self.core_tensor = frozen_core

        # copy the batchnorms of the original TuckER model and keep them frozen
        self.batch_norm_1 = copy.deepcopy(
            self.model.batch_norm_1
        )  # copy weights and stuff
        self.batch_norm_2 = copy.deepcopy(
            self.model.batch_norm_2
        )  # copy weights and stuff
        self.batch_norm_1.weight.requires_grad = False
        self.batch_norm_1.bias.requires_grad = False
        self.batch_norm_2.weight.requires_grad = False
        self.batch_norm_2.bias.requires_grad = False
        self.batch_norm_1.eval()
        self.batch_norm_2.eval()

    # Override
    def predict_triples(self, triples: np.array, original_mode: bool = False):
        """
        This method overrides the Model predict_triples method
        by adding the possibility to run predictions in original_mode
        which means ...
        :param triples: the DIRECT triples. Will be inverted to perform head prediction
        :param original_mode:
        :return:
        """

        direct_triples = triples

        # assert all triples are direct
        assert (triples[:, 1] < self.dataset.num_relations).all()

        # if we are in original_mode, make sure that the kelpie entity is not featured in the triples to predict
        # otherwise, make sure that the original entity is not featured in the triples to predict
        forbidden_entity_id = (
            self.kelpie_entity_id if original_mode else self.original_entity_id
        )
        assert np.isin(forbidden_entity_id, direct_triples[:][0, 2]) == False

        # use the TuckER implementation method to obtain scores, ranks and prediction results.
        # these WILL feature the forbidden entity, so we now need to filter them
        scores, ranks, predictions = TuckER.predict_triples(self, direct_triples)

        # remove any reference to the forbidden entity id
        # (that may have been included in the predicted entities)
        for i in range(len(direct_triples)):
            head_predictions, tail_predictions = predictions[i]
            head_rank, tail_rank = ranks[i]

            # remove the forbidden entity id from the head predictions (note: it could be absent due to filtering)
            # and if it was before the head target decrease the head rank by 1
            forbidden_indices = np.where(head_predictions == forbidden_entity_id)[0]
            if len(forbidden_indices) > 0:
                index = forbidden_indices[0]
                head_predictions = np.concatenate(
                    [head_predictions[:index], head_predictions[index + 1 :]], axis=0
                )
                if index < head_rank:
                    head_rank -= 1

            # remove the kelpie entity id from the tail predictions  (note: it could be absent due to filtering)
            # and if it was before the tail target decrease the head rank by 1
            forbidden_indices = np.where(tail_predictions == forbidden_entity_id)[0]
            if len(forbidden_indices) > 0:
                index = forbidden_indices[0]
                tail_predictions = np.concatenate(
                    [tail_predictions[:index], tail_predictions[index + 1 :]], axis=0
                )
                if index < tail_rank:
                    tail_rank -= 1

            predictions[i] = (head_predictions, tail_predictions)
            ranks[i] = (head_rank, tail_rank)

        return scores, ranks, predictions

    # Override
    def predict_triple(self, triple: np.array, original_mode: bool = False):
        """
        Override the
        :param triple: the DIRECT triple. Will be inverted to perform head prediction
        :param original_mode:
        :return:
        """

        assert triple[1] < self.dataset.num_relations

        scores, ranks, predictions = self.predict_triples(
            np.array([triple]), original_mode
        )
        return scores[0], ranks[0], predictions[0]
