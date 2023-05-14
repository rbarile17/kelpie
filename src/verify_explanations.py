import argparse
import copy
import random

import numpy
import torch

from . import ALL_DATASET_NAMES
from .data import MANY_TO_ONE, ONE_TO_ONE
from .data import Dataset
from .link_prediction.optimization import MultiClassNLLOptimizer
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

datasets = ALL_DATASET_NAMES

parser = argparse.ArgumentParser(
    description="Model-agnostic tool for explaining link predictions"
)

parser.add_argument(
    "--dataset", choices=datasets, help="Dataset in {}".format(datasets), required=True
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

parser.add_argument("--dimension", default=1000, type=int, help="Factorization rank.")

parser.add_argument("--learning_rate", default=1e-1, type=float, help="Learning rate")

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

args = parser.parse_args()

# deterministic!
seed = 42
numpy.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.cuda.set_rng_state(torch.cuda.get_rng_state())
torch.backends.cudnn.deterministic = True

#############  LOAD DATASET
# load the dataset and its training triples
print("Loading dataset %s..." % args.dataset)
dataset = Dataset(dataset=args.dataset)

# get the ids of the elements of the fact to explain and the perspective entity
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

with open("output.txt", "r") as input_file:
    input_lines = input_file.readlines()

original_model = ComplEx(
    dataset=dataset, hyperparameters=hyperparameters, init_random=True
)  # type: ComplEx
original_model.to("cuda")
original_model.load_state_dict(torch.load(args.model_path))
original_model.eval()

facts_to_explain = []
triples_to_explain = []
perspective = "head"  # for all triples the perspective was head for simplicity
triple_to_explain_2_best_rule = {}

if args.mode == "sufficient":
    triple_to_explain_2_entities_to_convert = {}

    i = 0
    while i <= len(input_lines) - 4:
        fact_line = input_lines[i]
        similar_entities_line = input_lines[i + 1]
        rules_line = input_lines[i + 2]
        empty_line = input_lines[i + 3]

        # triple to explain
        fact = tuple(fact_line.strip().split(";"))
        facts_to_explain.append(fact)
        triple = (
            dataset.entity_to_id[fact[0]],
            dataset.relation_to_id[fact[1]],
            dataset.entity_to_id[fact[2]],
        )
        triples_to_explain.append(triple)

        # similar entities
        similar_entities_names = similar_entities_line.strip().split("|")
        similar_entities = [dataset.entity_to_id[x] for x in similar_entities_names]
        triple_to_explain_2_entities_to_convert[triple] = similar_entities

        # rules
        rules_with_relevance = []

        rule_relevance_inputs = rules_line.strip().split("|")
        best_rule, best_rule_relevance_str = rule_relevance_inputs[0].split("::")
        best_rule_bits = best_rule.split(";")

        best_rule_facts = []
        j = 0
        while j < len(best_rule_bits):
            cur_head_name = best_rule_bits[j]
            cur_rel_name = best_rule_bits[j + 1]
            cur_tail_name = best_rule_bits[j + 2]

            best_rule_facts.append((cur_head_name, cur_rel_name, cur_tail_name))
            j += 3

        best_rule_triples = [dataset.ids_triple(x) for x in best_rule_facts]
        relevance = float(best_rule_relevance_str)
        rules_with_relevance.append((best_rule_triples, relevance))

        triple_to_explain_2_best_rule[triple] = best_rule_triples
        i += 4

    triples_to_add = []  # the triples to add to the training set before retraining
    triples_to_convert = (
        []
    )  # the triples that, after retraining, should have changed their predictions

    # for each triple to explain, get the corresponding similar entities and the most relevant triple in addition.
    # For each of those similar entities create
    #   - a version of the triple to explain that features the similar entity instead of the entity to explain
    #   - a version of the most relevant triple to add that features the similar entity instead of the entity to explain

    triple_to_convert_2_original_triple_to_explain = {}
    triples_to_convert_2_added_triples = {}
    for triple_to_explain in triples_to_explain:
        entity_to_explain = (
            triple_to_explain[0] if perspective == "head" else triple_to_explain[2]
        )

        cur_entities_to_convert = triple_to_explain_2_entities_to_convert[
            triple_to_explain
        ]

        cur_best_rule_triples = triple_to_explain_2_best_rule[triple_to_explain]

        for cur_entity_to_convert in cur_entities_to_convert:
            cur_triple_to_convert = Dataset.replace_entity_in_triple(
                triple=triple_to_explain,
                old_entity=entity_to_explain,
                new_entity=cur_entity_to_convert,
                as_numpy=False,
            )
            cur_triples_to_add = Dataset.replace_entity_in_triples(
                triples=cur_best_rule_triples,
                old_entity=entity_to_explain,
                new_entity=cur_entity_to_convert,
                as_numpy=False,
            )

            triples_to_convert.append(cur_triple_to_convert)
            triples_to_convert_2_added_triples[
                cur_triple_to_convert
            ] = cur_triples_to_add

            for cur_triple_to_add in cur_triples_to_add:
                triples_to_add.append(cur_triple_to_add)

            triple_to_convert_2_original_triple_to_explain[
                tuple(cur_triple_to_convert)
            ] = triple_to_explain

    new_dataset = copy.deepcopy(dataset)

    # if any of the triples_to_add overlaps contradicts any pre-existing facts
    # (e.g. adding "<Obama, born_in, Paris>" when the dataset already contains "<Obama, born_in, Honolulu>")
    # we need to remove such pre-eisting facts before adding the new triples_to_add
    print("Adding triples: ")
    for head, relation, tail in triples_to_add:
        print("\t" + dataset.printable_triple((head, relation, tail)))
        if new_dataset.relation_to_type[relation] in [MANY_TO_ONE, ONE_TO_ONE]:
            for pre_existing_tail in new_dataset.train_to_filter[(head, relation)]:
                new_dataset.remove_training_triple(
                    (head, relation, pre_existing_tail)
                )

    # append the triples_to_add to training triples of new_dataset
    # (and also update new_dataset.to_filter accordingly)
    new_dataset.add_training_triples(numpy.array(triples_to_add))

    # obtain tail ranks and scores of the original model for that all_triples_to_convert
    (
        original_scores,
        original_ranks,
        original_predictions,
    ) = original_model.predict_triples(numpy.array(triples_to_convert))

    new_model = ComplEx(
        dataset=new_dataset, hyperparameters=hyperparameters, init_random=True
    )  # type: ComplEx
    new_optimizer = MultiClassNLLOptimizer(
        model=new_model, hyperparameters=hyperparameters
    )
    new_optimizer.train(training_triples=new_dataset.training_triples)
    new_model.eval()

    new_scores, new_ranks, new_predictions = new_model.predict_triples(
        numpy.array(triples_to_convert)
    )

    for i in range(len(triples_to_convert)):
        cur_triple = triples_to_convert[i]
        original_direct_score = original_scores[i][0]
        original_tail_rank = original_ranks[i][1]

        new_direct_score = new_scores[i][0]
        new_tail_rank = new_ranks[i][1]

        print(
            "<"
            + ", ".join(
                [
                    dataset.id_to_entity[cur_triple[0]],
                    dataset.id_to_relation[cur_triple[1]],
                    dataset.id_to_entity[cur_triple[2]],
                ]
            )
            + ">"
        )
        print(
            "\tDirect score: from "
            + str(original_direct_score)
            + " to "
            + str(new_direct_score)
        )
        print(
            "\tTail rank: from " + str(original_tail_rank) + " to " + str(new_tail_rank)
        )
        print()

    output_lines = []
    for i in range(len(triples_to_convert)):
        cur_triple_to_convert = triples_to_convert[i]
        cur_added_triples = triples_to_add[i]
        original_triple_to_explain = triple_to_convert_2_original_triple_to_explain[
            tuple(cur_triple_to_convert)
        ]

        original_direct_score = original_scores[i][0]
        original_tail_rank = original_ranks[i][1]

        new_direct_score = new_scores[i][0]
        new_tail_rank = new_ranks[i][1]

        # original_head, original_relation, original_tail,
        # fact_to_convert_head, fact_to_convert_rel, fact_to_convert_tail,

        # original_direct_score, new_direct_score,
        #  original_tail_rank, new_tail_rank

        a = ";".join(dataset.labels_triple(original_triple_to_explain))
        b = ";".join(dataset.labels_triple(cur_triple_to_convert))

        c = []
        triples_to_add_to_this_entity = triples_to_convert_2_added_triples[
            cur_triple_to_convert
        ]
        for x in range(4):
            if x < len(triples_to_add_to_this_entity):
                c.append(
                    ";".join(dataset.labels_triple(triples_to_add_to_this_entity[x]))
                )
            else:
                c.append(";;")

        c = ";".join(c)
        d = str(original_direct_score) + ";" + str(new_direct_score)
        e = str(original_tail_rank) + ";" + str(new_tail_rank)
        output_lines.append(";".join([a, b, c, d, e]) + "\n")

    with open("output_end_to_end.csv", "w") as outfile:
        outfile.writelines(output_lines)


elif args.mode == "necessary":
    i = 0
    while i <= len(input_lines) - 3:
        fact_line = input_lines[i]
        rules_line = input_lines[i + 1]
        empty_line = input_lines[i + 2]

        # triple to explain
        fact = tuple(fact_line.strip().split(";"))
        facts_to_explain.append(fact)
        triple = (
            dataset.entity_to_id[fact[0]],
            dataset.relation_to_id[fact[1]],
            dataset.entity_to_id[fact[2]],
        )
        triples_to_explain.append(triple)

        # rules
        if rules_line.strip() != "":
            rules_with_relevance = []

            rule_relevance_inputs = rules_line.strip().split("|")
            best_rule, best_rule_relevance_str = rule_relevance_inputs[0].split("::")
            best_rule_bits = best_rule.split(";")

            best_rule_facts = []
            j = 0
            while j < len(best_rule_bits):
                cur_head_name = best_rule_bits[j]
                cur_rel_name = best_rule_bits[j + 1]
                cur_tail_name = best_rule_bits[j + 2]

                best_rule_facts.append((cur_head_name, cur_rel_name, cur_tail_name))
                j += 3

            best_rule_triples = [dataset.ids_triple(x) for x in best_rule_facts]

            if best_rule_relevance_str.startswith("["):
                best_rule_relevance_str = best_rule_relevance_str[1:]
            if best_rule_relevance_str.endswith("]"):
                best_rule_relevance_str = best_rule_relevance_str[:-1]
            relevance = float(best_rule_relevance_str)

            rules_with_relevance.append((best_rule_triples, relevance))

            triple_to_explain_2_best_rule[triple] = best_rule_triples
        else:
            triple_to_explain_2_best_rule[triple] = []

        i += 3

    triples_to_remove = (
        []
    )  # the triples to remove from the training set before retraining

    for triple_to_explain in triples_to_explain:
        best_rule_triples = triple_to_explain_2_best_rule[triple_to_explain]
        triples_to_remove += best_rule_triples

    new_dataset = copy.deepcopy(dataset)

    print("Removing triples: ")
    for head, relation, tail in triples_to_remove:
        print("\t" + dataset.printable_triple((head, relation, tail)))

    # remove the triples_to_remove from training triples of new_dataset (and update new_dataset.to_filter accordingly)
    new_dataset.remove_training_triples(triples_to_remove)

    # obtain tail ranks and scores of the original model for all triples_to_explain
    (
        original_scores,
        original_ranks,
        original_predictions,
    ) = original_model.predict_triples(numpy.array(triples_to_explain))

    ######

    new_model = ComplEx(
        dataset=new_dataset, hyperparameters=hyperparameters, init_random=True
    )  # type: ComplEx
    new_optimizer = MultiClassNLLOptimizer(
        model=new_model, hyperparameters=hyperparameters
    )
    new_optimizer.train(training_triples=new_dataset.training_triples)
    new_model.eval()

    new_scores, new_ranks, new_predictions = new_model.predict_triples(
        numpy.array(triples_to_explain)
    )

    for i in range(len(triples_to_explain)):
        cur_triple = triples_to_explain[i]
        original_direct_score = original_scores[i][0]
        original_tail_rank = original_ranks[i][1]

        new_direct_score = new_scores[i][0]
        new_tail_rank = new_ranks[i][1]

        print(
            "<"
            + ", ".join(
                [
                    dataset.id_to_entity[cur_triple[0]],
                    dataset.id_to_relation[cur_triple[1]],
                    dataset.id_to_entity[cur_triple[2]],
                ]
            )
            + ">"
        )
        print(
            "\tDirect score: from "
            + str(original_direct_score)
            + " to "
            + str(new_direct_score)
        )
        print(
            "\tTail rank: from " + str(original_tail_rank) + " to " + str(new_tail_rank)
        )
        print()

    output_lines = []
    for i in range(len(triples_to_explain)):
        cur_triple_to_explain = triples_to_explain[i]

        original_direct_score = original_scores[i][0]
        original_tail_rank = original_ranks[i][1]

        new_direct_score = new_scores[i][0]
        new_tail_rank = new_ranks[i][1]

        # original_head, original_relation, original_tail,
        # fact_to_convert_head, fact_to_convert_rel, fact_to_convert_tail,

        # original_direct_score, new_direct_score,
        #  original_tail_rank, new_tail_rank

        a = ";".join(dataset.labels_triple(cur_triple_to_explain))

        b = []
        triples_to_remove_from_this_entity = triple_to_explain_2_best_rule[
            cur_triple_to_explain
        ]
        for x in range(4):
            if x < len(triples_to_remove_from_this_entity):
                b.append(
                    ";".join(
                        dataset.labels_triple(triples_to_remove_from_this_entity[x])
                    )
                )
            else:
                b.append(";;")

        b = ";".join(b)
        c = str(original_direct_score) + ";" + str(new_direct_score)
        d = str(original_tail_rank) + ";" + str(new_tail_rank)
        output_lines.append(";".join([a, b, c, d]) + "\n")

    with open("output_end_to_end.csv", "w") as outfile:
        outfile.writelines(output_lines)
