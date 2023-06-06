import os
import argparse
import numpy
import torch

from .. import ALL_DATASET_NAMES
from ..config import MODEL_PATH
from ..data import Dataset

from .evaluation import Evaluator
from .optimization import BCEOptimizer, MultiClassNLLOptimizer, PairwiseRankingOptimizer
from .models import ComplEx, ConvE, TransE
from .models import (
    OPTIMIZER_NAME,
    LEARNING_RATE,
    REGULARIZER_NAME,
    REGULARIZER_WEIGHT,
    BATCH_SIZE,
    DECAY,
    DECAY_1,
    DECAY_2,
    EPOCHS,
    DIMENSION,
    HIDDEN_LAYER_SIZE,
    INIT_SCALE,
    INPUT_DROPOUT,
    FEATURE_MAP_DROPOUT,
    HIDDEN_DROPOUT,
    LABEL_SMOOTHING,
    MARGIN,
    NEGATIVE_TRIPLES_RATIO,
)

parser = argparse.ArgumentParser(description="Kelpie")

parser.add_argument(
    "--dataset",
    choices=ALL_DATASET_NAMES,
    help="Dataset in {}".format(ALL_DATASET_NAMES),
)

parser.add_argument(
    "--model",
    choices=["ConvE", "ComplEx", "TransE"],
    help=f"Model in {['ConvE', 'ComplEx', 'TransE']}",
)

optimizers = ["Adagrad", "Adam", "SGD"]
parser.add_argument(
    "--optimizer",
    choices=optimizers,
    default="Adagrad",
    help="Optimizer in {}".format(optimizers),
)

parser.add_argument("--max_epochs", default=50, type=int, help="Number of epochs.")

parser.add_argument(
    "--valid", default=-1, type=float, help="Number of epochs before valid."
)

parser.add_argument("--dimension", default=1000, type=int, help="Embedding dimension")

parser.add_argument(
    "--batch_size",
    default=1000,
    type=int,
    help="Number of triples in each mini-batch in SGD, Adagrad and Adam optimization",
)

parser.add_argument("--reg", default=0, type=float, help="Regularization weight")

parser.add_argument("--init_scale", default=1e-3, type=float, help="Initial scale")

parser.add_argument("--learning_rate", default=1e-1, type=float, help="Learning rate")
parser.add_argument("--decay_rate", type=float, default=1.0, help="Decay rate.")

parser.add_argument(
    "--input_dropout", type=float, default=0.3, nargs="?", help="Input layer dropout."
)

parser.add_argument(
    "--hidden_dropout", type=float, default=0.4, help="Dropout after the hidden layer."
)

parser.add_argument(
    "--feature_map_dropout",
    type=float,
    default=0.5,
    help="Dropout after the convolutional layer.",
)

parser.add_argument(
    "--label_smoothing", type=float, default=0.1, help="Amount of label smoothing."
)

parser.add_argument(
    "--hidden_size",
    type=int,
    default=9728,
    help="The side of the hidden layer. "
    "The required size changes with the size of the embeddings. Default: 9728 (embedding size 200).",
)

parser.add_argument(
    "--decay1",
    default=0.9,
    type=float,
    help="Decay rate for the first moment estimate in Adam",
)
parser.add_argument(
    "--decay2",
    default=0.999,
    type=float,
    help="Decay rate for second moment estimate in Adam",
)

parser.add_argument(
    "--margin", type=int, default=5, help="Margin for pairwise ranking loss."
)

parser.add_argument(
    "--negative_samples_ratio",
    type=int,
    default=3,
    help="Number of negative samples for each positive sample.",
)

parser.add_argument(
    "--regularizer_weight",
    type=float,
    default=0.0,
    help="Weight for L2 regularization.",
)

parser.add_argument("--load", help="path to the model to load", required=False)

args = parser.parse_args()

# deterministic!
seed = 42
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.set_rng_state(torch.cuda.get_rng_state())
torch.backends.cudnn.deterministic = True

if args.load is not None:
    model_path = args.load
else:
    model_path = os.path.join(MODEL_PATH, "_".join([args.model, args.dataset]) + ".pt")
    if not os.path.isdir(MODEL_PATH):
        os.makedirs(MODEL_PATH)

print("Loading %s dataset..." % args.dataset)
dataset = Dataset(dataset=args.dataset)

print("Initializing model...")
if args.model == "ComplEx":
    hyperparameters = {
        DIMENSION: args.dimension,
        INIT_SCALE: args.init_scale,
        OPTIMIZER_NAME: args.optimizer,
        BATCH_SIZE: args.batch_size,
        EPOCHS: args.max_epochs,
        LEARNING_RATE: args.learning_rate,
        DECAY_1: args.decay1,
        DECAY_2: args.decay2,
        REGULARIZER_NAME: "N3",
        REGULARIZER_WEIGHT: args.reg,
    }

    model = ComplEx(dataset=dataset, hyperparameters=hyperparameters, init_random=True)
    optimizer = MultiClassNLLOptimizer(model=model, hyperparameters=hyperparameters)
elif args.model == "TransE":
    hyperparameters = {
        DIMENSION: args.dimension,
        MARGIN: args.margin,
        NEGATIVE_TRIPLES_RATIO: args.negative_samples_ratio,
        REGULARIZER_WEIGHT: args.regularizer_weight,
        BATCH_SIZE: args.batch_size,
        LEARNING_RATE: args.learning_rate,
        EPOCHS: args.max_epochs,
    }

    model = TransE(dataset=dataset, hyperparameters=hyperparameters, init_random=True)
    optimizer = PairwiseRankingOptimizer(model=model, hyperparameters=hyperparameters)
elif args.model == "ConvE":
    hyperparameters = {
        DIMENSION: args.dimension,
        INPUT_DROPOUT: args.input_dropout,
        FEATURE_MAP_DROPOUT: args.feature_map_dropout,
        HIDDEN_DROPOUT: args.hidden_dropout,
        HIDDEN_LAYER_SIZE: args.hidden_size,
        BATCH_SIZE: args.batch_size,
        LEARNING_RATE: args.learning_rate,
        DECAY: args.decay_rate,
        LABEL_SMOOTHING: args.label_smoothing,
        EPOCHS: args.max_epochs,
    }

    model = ConvE(dataset=dataset, hyperparameters=hyperparameters, init_random=True)
    optimizer = BCEOptimizer(model=model, hyperparameters=hyperparameters)


model.to("cuda")
if args.load is not None:
    model.load_state_dict(torch.load(model_path))

print("Training model...")

optimizer.train(
    training_triples=dataset.training_triples,
    save_path=model_path,
    evaluate_every=args.valid,
    valid_triples=dataset.validation_triples,
)

print("Evaluating model...")
mrr, h1, h10, mr = Evaluator(model=model).evaluate(
    triples=dataset.testing_triples, write_output=False
)
print("\tTest Hits@1: %f" % h1)
print("\tTest Hits@10: %f" % h10)
print("\tTest Mean Reciprocal Rank: %f" % mrr)
print("\tTest Mean Rank: %f" % mr)
