import argparse
import torch

from .. import ALL_DATASET_NAMES
from ..data import Dataset

from .evaluation import Evaluator
from .models import ConvE, ComplEx, TransE
from .models import (
    LEARNING_RATE,
    REGULARIZER_WEIGHT,
    BATCH_SIZE,
    DECAY,
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

parser.add_argument("--max_epochs", type=int, default=1000, help="Number of epochs.")

parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")

parser.add_argument(
    "--learning_rate", type=float, default=0.0005, help="Learning rate."
)
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
    "--dimension", type=int, default=200, help="Embedding dimensionality."
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

parser.add_argument("--init_scale", default=1e-3, type=float, help="Initial scale")

parser.add_argument("--model_path", help="path to the model to load", required=True)

args = parser.parse_args()

print("Loading %s dataset..." % args.dataset)
dataset = Dataset(dataset=args.dataset)

print("Initializing model...")
if args.model == "ComplEx":
    hyperparameters = {DIMENSION: args.dimension, INIT_SCALE: args.init_scale}
    model = ComplEx(dataset=dataset, hyperparameters=hyperparameters, init_random=True)

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

model.to("cuda")
model.load_state_dict(torch.load(args.model_path))
model.eval()

print("Evaluating model...")
mrr, h1, h10, mr = Evaluator(model=model).evaluate(
    triples=dataset.testing_triples, write_output=True
)
print("\tTest Hits@1: %f" % h1)
print("\tTest Hits@10: %f" % h10)
print("\tTest Mean Reciprocal Rank: %f" % mrr)
print("\tTest Mean Rank: %f" % mr)
