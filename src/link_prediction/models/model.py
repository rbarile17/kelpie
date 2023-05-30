import numpy
import torch

from typing import Any, Tuple

from torch import nn
from ...data import Dataset

# KEYS FOR SUPPORTED HYPERPARAMETERS (to use in hyperparameter dicts)
DIMENSION = "dimension"  # embedding dimension, when both entity and relation embeddings have same dimension
ENTITY_DIMENSION = "entity_dimension"  # entity embedding dimension, when entity and relation embeddings have different dimensions
RELATION_DIMENSION = "relation_dimension"  # relation embedding dimension, when entity and relation embeddings have different dimensions

INPUT_DROPOUT = "input_dropout"  # dropout rate for the input embeddings
HIDDEN_DROPOUT = "hidden_dropout"  # dropout rate after the hidden layer when there is only one hidden layer
HIDDEN_DROPOUT_1 = "hidden_dropout_1"  # dropout rate after the first hidden layer
HIDDEN_DROPOUT_2 = "hidden_dropout_2"  # dropout rate after the second hidden layer
FEATURE_MAP_DROPOUT = "feature_map_dropout"  # feature map dropout

HIDDEN_LAYER_SIZE = (
    "hidden_layer"  # hidden layer size when there is only one hidden layer
)

INIT_SCALE = (
    "init_scale"  # downscale to operate on the initial, randomly generated embeddings
)
OPTIMIZER_NAME = (
    "optimizer_name"  # name of the optimization technique: Adam, Adagrad, SGD
)
BATCH_SIZE = "batch_size"  # training batch size
EPOCHS = "epochs"  # training epochs
LEARNING_RATE = "learning_rate"  # learning rate
DECAY = "decay"  #
DECAY_1 = "decay_1"  # Adam decay 1
DECAY_2 = "decay_2"  # Adam decay 2

MARGIN = "margin"  # pairwise margin-based loss margin
NEGATIVE_TRIPLES_RATIO = "negative_triples"  # number of negative triples to obtain, via corruption, for each positive triple

REGULARIZER_NAME = "regularizer"  # name of the regularization technique: N3
REGULARIZER_WEIGHT = "regularizer_weight"  # weight for the regularization in the loss
LABEL_SMOOTHING = "label_smoothing"  # label smoothing value

GAMMA = "gamma"


class Model(nn.Module):
    """
    The Model class provides the interface that any LP model should implement.

    The responsibility of any Model implementation is to
        - store the embeddings for entities and relations
        - implement the specific scoring function for that link prediction model
        - offer methods that use that scoring function
            either to run prediction on one or multiple triples, or to run forward propagation in training

    On the contrary, training and evaluation are not performed directly by the model class,
    but require the use of an Optimizer or an Evaluator object respectively.

    All models work with entity and relation ids directly (not with their names),
    that they found from the Dataset object used to initialize the Model.

    Whenever a Model method requires triples, it accepts them in the form of 2-dimensional numpy.arrays,
    where each row corresponds to a triple and contains the integer ids of its head, relation and tail.
    """

    def __init__(self, dataset: Dataset):
        nn.Module.__init__(self)
        self.dataset = dataset

    def is_minimizer(self):
        """
        This method specifies whether this model aims at minimizing of maximizing scores.
        :return: True if in this model low scores are better than high scores; False otherwise.
        """
        pass

    def score(self, triples: numpy.array) -> numpy.array:
        """
        This method computes and returns the plausibility scores for a collection of triples.

        :param triples: a numpy array containing all the triples to score
        :return: the computed scores, as a numpy array
        """
        pass

    # override
    def all_scores(self, triples: numpy.array):
        """
        This method computes, For each of the passed triples, the score for all possible tail entities.
        :param triples: a 2-dimensional numpy array containing the triples to score, one per row
        :return: a 2-dimensional numpy array that, for each triple, contains a row for each passed triple
                 and a column for each possible tail
        """

        pass

    def forward(self, triples: numpy.array):
        """
        This method performs forward propagation for a collection of triples.
        This method is only used in training, when an Optimizer calls it passing the current batch of triples.

        This method returns all the items needed by the Optimizer to perform gradient descent in this training step.
        Such items heavily depend on the specific Model implementation;
        they usually include the scores for the triples (in a form usable by the ML framework, e.g. torch.Tensors)
        but may also include other stuff (e.g. the involved embeddings themselves, that the Optimizer
        may use to compute regularization factors)

        :param triples: a numpy array containing all the triples to perform forward propagation on
        """
        pass

    def predict_triples(self, triples: numpy.array) -> Tuple[Any, Any, Any]:
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
        pass

    def predict_triple(self, triple: numpy.array) -> Tuple[Any, Any, Any]:
        """
        This method performs prediction on one (direct) triple, and returns the corresponding
        score, ranks and prediction lists.

        :param triple: the triple to predict, as a numpy array.
        :return: this method returns 3 items:
                - the triple score
                         OR IF THE MODEL SUPPORTS INVERSE FACTS
                  a couple containing the scores of the triple and of its inverse

                - a couple containing the head rank and the tail rank

                - a couple containing the head_predictions and tail_predictions numpy arrays;
                    > head_predictions contains all entities predicted as heads, sorted by decreasing plausibility
                    [NB: the target head will be in this numpy array in position head_rank-1]
                    > tail_predictions contains all entities predicted as tails, sorted by decreasing plausibility
                    [NB: the target tail will be in this numpy array in position tail_rank-1]
        """

        assert triple[1] < self.dataset.num_relations

        [result] = self.predict_triples(numpy.array([triple]))
        return result["score"], result["rank"], result["prediction"]

    def kelpie_model_class(self):
        """
        This method provides the KelpieModel implementation class corresponding to this specific Model class.
        E.g. ComplEx.kelpie_model_class() -> KelpieComplEx.__class__

        When called on a KelpieModel subclass, it raises an exception.

        :return: The KelpieModel class that corresponds to the Model Class running this method
        :raise: an Exception if called on a KelpieModel subclass
        """
        pass


class KelpieModel(Model):
    """
    The KelpieModel class provides the interface that any post-trainable LP model should implement.

    The main functions of KelpieModel are thus identical to Model
    (which is why KelpieModel extends Model).

    In addition to that, a KelpieModels also
    """

    # override
    def predict_triples(self, triples: numpy.array, original_mode: bool = False):
        """
        This method interface overrides the superclass method by adding the option to run predictions in original_mode,
        which means ignoring in any circumstances the additional "fake" kelpie entity.

        :param triples: the DIRECT triples. They will be inverted to perform head prediction
        :param original_mode: a boolean flag specifying whether to work in original_mode or to use the kelpie entity

        :return: a numpy array containing
        """
        pass

    # Override
    def predict_triple(self, triple: numpy.array, original_mode: bool = False):
        """
        This method interface overrides the superclass method by adding the option to run predictions in original_mode,
        which means ignoring in any circumstances the additional "fake" kelpie entity.

        :param triple: the DIRECT triple. It will be inverted to perform head prediction
        :param original_mode: a boolean flag specifying whether to work in original_mode or to use the kelpie entity

        :return:
        """

    # this is necessary
    def update_embeddings(self):
        with torch.no_grad():
            self.entity_embeddings[self.kelpie_entity_id] = self.kelpie_entity_embedding

    # override
    def train(self, mode=True):
        """
        This method overrides the traditional train() implementation of torch.nn.Module,
        in which a.train() sets all children of a to train mode.

        In KelpieModels, in post-training any layers, including BatchNorm1d or BatchNorm2d,
        must NOT be put in train mode, because even in post-training they MUST remain constant.

        So this method overrides the traditional train() by skipping any layer children
        :param mode:
        :return:
        """
        self.training = mode
        for module in self.children():
            if not (
                isinstance(module, Model)
                or isinstance(module, torch.nn.BatchNorm1d)
                or isinstance(module, torch.nn.BatchNorm2d)
                or isinstance(module, torch.nn.Linear)
                or isinstance(module, torch.nn.Conv2d)
            ):
                module.train(mode)
        return self

    def kelpie_model_class(self):
        raise Exception(self.__class__.name + " is a KelpieModel.")
