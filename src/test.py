import argparse
import torch

from .data import ALL_DATASET_NAMES
from .data import Dataset

from .link_prediction.evaluation import Evaluator
from .link_prediction.models import ComplEx
from .link_prediction.models import DIMENSION, INIT_SCALE

parser = argparse.ArgumentParser(description="Kelpie")

parser.add_argument(
    "--dataset",
    choices=ALL_DATASET_NAMES,
    help="Dataset in {}".format(ALL_DATASET_NAMES),
)

parser.add_argument("--dimension", default=1000, type=int, help="Embedding dimension")

parser.add_argument("--init_scale", default=1e-3, type=float, help="Initial scale")

parser.add_argument("--learning_rate", default=1e-1, type=float, help="Learning rate")

parser.add_argument("--model_path", help="path to the model to load", required=True)

args = parser.parse_args()

print("Loading %s dataset..." % args.dataset)
dataset = Dataset(dataset=args.dataset)

hyperparameters = {DIMENSION: args.dimension, INIT_SCALE: args.init_scale}
print("Initializing model...")
model = ComplEx(
    dataset=dataset, hyperparameters=hyperparameters, init_random=True
)  # type: ComplEx
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
