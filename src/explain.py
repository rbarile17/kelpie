import argparse
import json
import random
import time

import numpy
import torch

from . import ALL_DATASET_NAMES
from .data import Dataset
from .explanation_systems.kelpie import Kelpie
from .explanation_systems.data_poisoning import DataPoisoning
from .explanation_systems.criage import Criage
from .prefilters import (
    TOPOLOGY_PREFILTER,
    TYPE_PREFILTER,
    NO_PREFILTER,
    WEIGHTED_TOPOLOGY_PREFILTER,
)
from .link_prediction.models import ComplEx
from .link_prediction.models import (
    LEARNING_RATE,
    OPTIMIZER_NAME,
    DECAY_1,
    DECAY_2,
    REGULARIZER_WEIGHT,
    EPOCHS,
    BATCH_SIZE,
    REGULARIZER_NAME,
    DIMENSION,
    INIT_SCALE,
)


def parse_args():
    datasets = ALL_DATASET_NAMES

    parser = argparse.ArgumentParser(
        description="Model-agnostic tool for explaining link predictions"
    )

    parser.add_argument(
        "--dataset",
        choices=datasets,
        help="Dataset in {}".format(datasets),
        required=True,
    )

    optimizers = ["Adagrad", "Adam", "SGD"]
    parser.add_argument(
        "--optimizer",
        choices=optimizers,
        default="Adagrad",
        help="Optimizer in {} to use in post-training".format(optimizers),
    )

    parser.add_argument(
        "--batch_size", default=100, type=int, help="Batch size to use in post-training"
    )

    parser.add_argument(
        "--max_epochs",
        default=200,
        type=int,
        help="Number of epochs to run in post-training",
    )

    parser.add_argument(
        "--dimension", default=1000, type=int, help="Factorization rank."
    )

    parser.add_argument(
        "--learning_rate", default=1e-1, type=float, help="Learning rate"
    )

    parser.add_argument("--reg", default=0, type=float, help="Regularization weight")

    parser.add_argument("--init", default=1e-3, type=float, help="Initial scale")

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
        "--model_path",
        help="Path to the model to explain the predictions of",
        required=True,
    )

    parser.add_argument(
        "--triples_to_explain",
        type=str,
        required=True,
        help="path of the file with the facts to explain the predictions of.",
    )

    parser.add_argument(
        "--coverage",
        type=int,
        default=10,
        help="Number of random entities to extract and convert",
    )

    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        choices=[None, "k1", "data_poisoning", "criage"],
        help="attribute to use when we want to use a baseline rather than the Kelpie engine",
    )

    parser.add_argument(
        "--entities_to_convert",
        type=str,
        help="path of the file with the entities to convert (only used by baselines)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="sufficient",
        choices=["sufficient", "necessary"],
        help="The explanation mode",
    )

    parser.add_argument(
        "--relevance_threshold",
        type=float,
        default=None,
        help="The relevance acceptance threshold to use",
    )

    prefilters = [
        TOPOLOGY_PREFILTER,
        TYPE_PREFILTER,
        NO_PREFILTER,
        WEIGHTED_TOPOLOGY_PREFILTER,
    ]
    parser.add_argument(
        "--prefilter",
        choices=prefilters,
        default="graph-based",
        help="Prefilter type in {} to use in pre-filtering".format(prefilters),
    )

    parser.add_argument(
        "--prefilter_threshold",
        type=int,
        default=20,
        help="The number of promising training facts to keep after prefiltering",
    )

    return parser.parse_args()


def main(args):
    seed = 42
    torch.backends.cudnn.deterministic = True
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.set_rng_state(torch.cuda.get_rng_state())

    hyperparameters = {
        DIMENSION: args.dimension,
        INIT_SCALE: args.init,
        LEARNING_RATE: args.learning_rate,
        OPTIMIZER_NAME: args.optimizer,
        DECAY_1: args.decay1,
        DECAY_2: args.decay2,
        REGULARIZER_WEIGHT: args.reg,
        EPOCHS: args.max_epochs,
        BATCH_SIZE: args.batch_size,
        REGULARIZER_NAME: "N3",
    }

    prefilter = args.prefilter
    relevance_threshold = args.relevance_threshold

    print(f"Loading dataset {args.dataset}...")
    dataset = Dataset(dataset=args.dataset)

    print("Reading triples to explain...")
    with open(args.triples_to_explain, "r") as triples_file:
        triples_to_explain = [x.strip().split("\t") for x in triples_file.readlines()]

    model = ComplEx(dataset=dataset, hyperparameters=hyperparameters, init_random=True)
    model.to("cuda")
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    start_time = time.time()

    if args.baseline is None:
        kelpie = Kelpie(
            model=model,
            dataset=dataset,
            hyperparameters=hyperparameters,
            prefilter_type=prefilter,
            relevance_threshold=relevance_threshold,
        )
    elif args.baseline == "data_poisoning":
        kelpie = DataPoisoning(
            model=model,
            dataset=dataset,
            hyperparameters=hyperparameters,
            prefilter_type=prefilter,
        )
    elif args.baseline == "criage":
        kelpie = Criage(model=model, dataset=dataset, hyperparameters=hyperparameters)
    elif args.baseline == "k1":
        kelpie = Kelpie(
            model=model,
            dataset=dataset,
            hyperparameters=hyperparameters,
            prefilter_type=prefilter,
            relevance_threshold=relevance_threshold,
            max_explanation_length=1,
        )
    else:
        kelpie = Kelpie(
            model=model,
            dataset=dataset,
            hyperparameters=hyperparameters,
            prefilter_type=prefilter,
            relevance_threshold=relevance_threshold,
        )

    explanations = []
    for i, triple in enumerate(triples_to_explain):
        head, relation, tail = triple
        print(
            f"\nExplaining triple {i + 1} on {len(triples_to_explain)}: " \
            f"<{head},{relation},{tail}>"
        )
        triple = dataset.ids_triple(triple)

        if args.mode == "sufficient":
            (
                rule_to_relevance,
                entities_to_convert,
            ) = kelpie.explain_sufficient(
                triple_to_explain=triple,
                perspective="head",
                num_promising_triples=args.prefilter_threshold,
                num_entities_to_convert=args.coverage,
                entities_to_convert=None,
            )

            if entities_to_convert is None or len(entities_to_convert) == 0:
                continue
            entities_to_convert = [dataset.id_to_entity[x] for x in entities_to_convert]

            rule_to_relevance = [
                (
                    [dataset.labels_triple(rule_triple) for rule_triple in rule],
                    relevance,
                )
                for rule, relevance in rule_to_relevance
            ]

            explanations.append(
                {
                    "triple": dataset.labels_triple(triple),
                    "entities_to_convert": entities_to_convert,
                    "rule_to_relevance": rule_to_relevance,
                }
            )
        elif args.mode == "necessary":
            rule_to_relevance = kelpie.explain_necessary(
                triple_to_explain=triple,
                perspective="head",
                num_promising_triples=args.prefilter_threshold,
            )
            rule_to_relevance = [
                (
                    [dataset.labels_triple(rule_triple) for rule_triple in rule],
                    relevance,
                )
                for rule, relevance in rule_to_relevance
            ]
            explanations.append(
                {
                    "triple": dataset.labels_triple(triple),
                    "rule_to_relevance": rule_to_relevance,
                }
            )

    print("Required time: " + str(time.time() - start_time) + " seconds")
    with open("output.json", "w") as output:
        json.dump(explanations, output)


if __name__ == "__main__":
    main(parse_args())
