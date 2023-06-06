import argparse
import copy
import json
import random

import numpy
import torch

from collections import defaultdict

from . import ALL_DATASET_NAMES
from .data import MANY_TO_ONE, ONE_TO_ONE
from .data import Dataset
from .link_prediction.optimization import (
    BCEOptimizer,
    MultiClassNLLOptimizer,
    PairwiseRankingOptimizer,
)
from .link_prediction.models import ConvE, ComplEx, TransE
from .link_prediction.models import (
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

    parser.add_argument(
        "--model",
        choices=["ConvE", "ComplEx", "TransE"],
        help=f"Model in {['ConvE', 'ComplEx', 'TransE']}",
    )

    parser.add_argument(
        "--model_path",
        help="Path to the model to explain the predictions of",
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
    parser.add_argument("--decay_rate", type=float, default=1.0, help="Decay rate.")

    parser.add_argument(
        "--input_dropout",
        type=float,
        default=0.3,
        nargs="?",
        help="Input layer dropout.",
    )

    parser.add_argument(
        "--hidden_dropout",
        type=float,
        default=0.4,
        help="Dropout after the hidden layer.",
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
        "--mode",
        type=str,
        default="sufficient",
        choices=["sufficient", "necessary"],
        help="The explanation mode",
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

    return parser.parse_args()


def main(args):
    seed = 42
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.set_rng_state(torch.cuda.get_rng_state())
    torch.backends.cudnn.deterministic = True

    with open("output.json", "r") as input_file:
        explanations = json.load(input_file)

    print(f"Loading dataset {args.dataset}...")
    dataset = Dataset(dataset=args.dataset)

    if args.model == "ComplEx":
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
        model = ComplEx(
            dataset=dataset, hyperparameters=hyperparameters, init_random=True
        )
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
        model = TransE(
            dataset=dataset, hyperparameters=hyperparameters, init_random=True
        )
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

        model = ConvE(
            dataset=dataset, hyperparameters=hyperparameters, init_random=True
        )

    model.to("cuda")
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    triples_to_explain = []
    perspective = "head"
    triple_to_best_rule = {}

    if args.mode == "sufficient":
        triple_to_convert_set = {}
        for explanation in explanations:
            triple_to_explain = dataset.ids_triple(explanation["triple"])
            triples_to_explain.append(triple_to_explain)

            entities = [entity for entity in explanation["entities_to_convert"]]
            entities = [dataset.entity_to_id[entity] for entity in entities]
            triple_to_convert_set[triple_to_explain] = entities

            best_rule, _ = explanation["rule_to_relevance"][0]
            best_rule = [dataset.ids_triple(triple) for triple in best_rule]

            triple_to_best_rule[triple_to_explain] = best_rule

        triples_to_add = []
        triples_to_convert = []

        triple_to_convert_to_added = {}
        for triple_to_explain in triples_to_explain:
            head, _, tail = triple_to_explain
            entity_to_explain = head if perspective == "head" else tail
            entities = triple_to_convert_set[triple_to_explain]
            best_rule = triple_to_best_rule[triple_to_explain]

            cur_triples_to_convert = []
            for entity in entities:
                triple_to_convert = Dataset.replace_entity_in_triple(
                    triple=triple_to_explain,
                    old_entity=entity_to_explain,
                    new_entity=entity,
                )
                cur_triples_to_convert.append(triple_to_convert)
                cur_triples_to_add = Dataset.replace_entity_in_triples(
                    triples=best_rule,
                    old_entity=entity_to_explain,
                    new_entity=entity,
                )
                triples_to_add.extend(cur_triples_to_add)
                triple_to_convert_to_added[triple_to_convert] = cur_triples_to_add

            triples_to_convert.extend(cur_triples_to_convert)
            triple_to_convert_set[triple_to_explain] = cur_triples_to_convert

        new_dataset = copy.deepcopy(dataset)

        print("Adding triples: ")
        for head, relation, tail in triples_to_add:
            print(f"\t{dataset.printable_triple((head, relation, tail))}")
            if new_dataset.relation_to_type[relation] in [MANY_TO_ONE, ONE_TO_ONE]:
                for existing_tail in new_dataset.train_to_filter[(head, relation)]:
                    new_dataset.remove_training_triple((head, relation, existing_tail))

        new_dataset.add_training_triples(triples_to_add)

        results = model.predict_triples(numpy.array(triples_to_convert))
        results = {
            triple: result for triple, result in zip(triples_to_convert, results)
        }

        if args.model == "ComplEx":
            new_model = ComplEx(
                dataset=new_dataset, hyperparameters=hyperparameters, init_random=True
            )
            new_optimizer = MultiClassNLLOptimizer(
                model=new_model, hyperparameters=hyperparameters
            )
        elif args.model == "TransE":
            new_model = TransE(
                dataset=new_dataset, hyperparameters=hyperparameters, init_random=True
            )
            new_optimizer = PairwiseRankingOptimizer(
                model=new_model, hyperparameters=hyperparameters
            )
        elif args.model == "ConvE":
            new_model = ConvE(
                dataset=new_dataset, hyperparameters=hyperparameters, init_random=True
            )
            new_optimizer = BCEOptimizer(
                model=new_model, hyperparameters=hyperparameters
            )
        new_optimizer.train(training_triples=new_dataset.training_triples)
        new_model.eval()
        new_results = new_model.predict_triples(numpy.array(triples_to_convert))
        new_results = {
            triple: result for triple, result in zip(triples_to_convert, new_results)
        }

        evaluations = []
        for triple_to_explain in triples_to_explain:
            triples_to_convert = triple_to_convert_set[triple_to_explain]
            evaluation = {
                "triple_to_explain": dataset.labels_triple(triple_to_explain),
            }
            conversions = []
            for triple_to_explain in triples_to_convert:
                result = results[triple_to_explain]
                new_result = new_results[triple_to_explain]

                score = result["score"]["tail"]
                rank = result["rank"]["tail"]
                new_score = new_result["score"]["tail"]
                new_rank = new_result["rank"]["tail"]

                print(dataset.printable_triple(triple_to_explain))
                print(f"\tDirect score: from {str(score)} to {str(new_score)}")
                print(f"\tTail rank: from {str(rank)} to {str(new_rank)}")
                print()

                conversions.append(
                    {
                        "triples_to_add": [
                            dataset.labels_triple(triple)
                            for triple in triple_to_convert_to_added[triple_to_explain]
                        ],
                        "score": str(score),
                        "rank": str(rank),
                        "new_score": str(new_score),
                        "new_rank": str(new_rank),
                    }
                )
            evaluation["conversions"] = conversions
            evaluations.append(evaluation)

    elif args.mode == "necessary":
        triple_to_best_rule = defaultdict(list)
        for explanation in explanations:
            triple_to_explain = dataset.ids_triple(explanation["triple"])
            triples_to_explain.append(triple_to_explain)
            best_rule, _ = explanation["rule_to_relevance"][0]
            best_rule = [dataset.ids_triple(triple) for triple in best_rule]

            triple_to_best_rule[triple_to_explain] = best_rule

        triples_to_remove = []

        for triple_to_explain in triples_to_explain:
            triples_to_remove += triple_to_best_rule[triple_to_explain]

        new_dataset = copy.deepcopy(dataset)
        print("Removing triples: ")
        for head, relation, tail in triples_to_remove:
            print(f"\t{dataset.printable_triple((head, relation, tail))}")

        new_dataset.remove_training_triples(triples_to_remove)

        results = model.predict_triples(numpy.array(triples_to_explain))
        results = {
            triple: result for triple, result in zip(triples_to_explain, results)
        }
        if args.model == "ComplEx":
            new_model = ComplEx(
                dataset=new_dataset, hyperparameters=hyperparameters, init_random=True
            )
            new_optimizer = MultiClassNLLOptimizer(
                model=new_model, hyperparameters=hyperparameters
            )
        elif args.model == "TransE":
            new_model = TransE(
                dataset=new_dataset, hyperparameters=hyperparameters, init_random=True
            )
            new_optimizer = PairwiseRankingOptimizer(
                model=new_model, hyperparameters=hyperparameters
            )
        elif args.model == "ConvE":
            new_model = ConvE(
                dataset=new_dataset, hyperparameters=hyperparameters, init_random=True
            )
            new_optimizer = BCEOptimizer(
                model=new_model, hyperparameters=hyperparameters
            )
        new_optimizer.train(training_triples=new_dataset.training_triples)
        new_model.eval()

        new_results = new_model.predict_triples(numpy.array(triples_to_explain))
        new_results = {
            triple: result for triple, result in zip(triples_to_explain, new_results)
        }

        evaluations = []
        for triple_to_explain in triples_to_explain:
            result = results[triple_to_explain]
            new_result = new_results[triple_to_explain]

            score = result["score"]["tail"]
            rank = result["rank"]["tail"]
            new_score = new_result["score"]["tail"]
            new_rank = new_result["rank"]["tail"]

            print(dataset.printable_triple(triple_to_explain))
            print(f"\tDirect score: from {str(score)} to {str(new_score)}")
            print(f"\tTail rank: from {str(rank)} to {str(new_rank)}")
            print()

            evaluation = {
                "triple_to_explain": dataset.labels_triple(triple_to_explain),
                "rule": [
                    dataset.labels_triple(triple)
                    for triple in triple_to_best_rule[triple_to_explain]
                ],
                "score": str(score),
                "rank": str(rank),
                "new_score": str(new_score),
                "new_rank": str(new_rank),
            }

            evaluations.append(evaluation)

    with open("output_end_to_end.json", "w") as outfile:
        json.dump(evaluations, outfile, indent=4)


if __name__ == "__main__":
    main(parse_args())
