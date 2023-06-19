import copy
import click
import json

import numpy
import torch

from collections import defaultdict

from . import DATASETS, MODELS_PATH
from .data import MANY_TO_ONE, ONE_TO_ONE
from .link_prediction import MODEL_REGISTRY

from .data import Dataset
from .utils import set_seeds

modes = ["necessary", "sufficient"]


@click.command()
@click.option("--dataset", type=click.Choice(DATASETS))
@click.option(
    "--model_config",
    type=click.Path(exists=True),
    help="Path of the model config (.json or .yml).",
)
@click.option("--mode", type=click.Choice(modes))
def main(
    dataset,
    model_config,
    mode,
):
    set_seeds(42)

    with open("output.json", "r") as input_file:
        explanations = json.load(input_file)

    model_config = json.load(open(model_config, "r"))
    model = model_config["model"]
    model_path = model_config.get("model_path", MODELS_PATH / f"{model}_{dataset}.pt")

    print(f"Loading dataset {dataset}...")
    dataset = Dataset(dataset=dataset)

    print(f"Loading model {model}...")
    model_class = MODEL_REGISTRY[model]["class"]
    optimizer_class = MODEL_REGISTRY[model]["optimizer"]
    hyperparams_class = model_class.get_hyperparams_class()
    model_hp = hyperparams_class(**model_config["model_params"])
    model = model_class(dataset=dataset, hp=model_hp, init_random=True)
    model.to("cuda")

    model.load_state_dict(torch.load(model_path))
    model.eval()

    preds = []
    triple_to_best_rule = {}

    if mode == "sufficient":
        triple_to_convert_set = {}
        for explanation in explanations:
            pred = dataset.ids_triple(explanation["triple"])
            preds.append(pred)

            entities = explanation["entities_to_convert"]
            entities = [dataset.entity_to_id[entity] for entity in entities]
            triple_to_convert_set[pred] = entities

            best_rule, _ = explanation["rule_to_relevance"][0]
            best_rule = [dataset.ids_triple(triple) for triple in best_rule]

            triple_to_best_rule[pred] = best_rule

        triples_to_add = []
        triples_to_convert = []

        triple_to_convert_to_added = {}
        for pred in preds:
            s, _, o = pred
            entities = triple_to_convert_set[pred]
            best_rule = triple_to_best_rule[pred]

            cur_triples_to_convert = []
            for entity in entities:
                triple_to_convert = Dataset.replace_entity_in_triple(pred, s, entity)
                cur_triples_to_convert.append(triple_to_convert)
                cur_triples_to_add = Dataset.replace_entity_in_triples(
                    best_rule, s, entity
                )
                triples_to_add.extend(cur_triples_to_add)
                triple_to_convert_to_added[triple_to_convert] = cur_triples_to_add

            triples_to_convert.extend(cur_triples_to_convert)
            triple_to_convert_set[pred] = cur_triples_to_convert

        new_dataset = copy.deepcopy(dataset)

        print("Adding triples: ")
        for s, p, o in triples_to_add:
            print(f"\t{dataset.printable_triple((s, p, o))}")
            if new_dataset.relation_to_type[p] in [MANY_TO_ONE, ONE_TO_ONE]:
                for existing_o in new_dataset.train_to_filter[(s, p)]:
                    new_dataset.remove_training_triple((s, p, existing_o))

        new_dataset.add_training_triples(triples_to_add)

        results = model.predict_triples(numpy.array(triples_to_convert))
        results = {
            triple: result for triple, result in zip(triples_to_convert, results)
        }

        new_model = model_class(dataset=new_dataset, hp=model_hp, init_random=True)
        hp = model_config["training"]
        optimizer_params = optimizer_class.get_hyperparams_class()(**hp)
        optimizer = optimizer_class(model=model, hp=optimizer_params, verbose=False)

        optimizer.train(training_triples=new_dataset.training_triples)
        new_model.eval()
        new_results = new_model.predict_triples(numpy.array(triples_to_convert))
        new_results = {
            triple: result for triple, result in zip(triples_to_convert, new_results)
        }

        evaluations = []
        for pred in preds:
            triples_to_convert = triple_to_convert_set[pred]
            evaluation = {
                "triple_to_explain": dataset.labels_triple(pred),
            }
            conversions = []
            for pred in triples_to_convert:
                result = results[pred]
                new_result = new_results[pred]

                score = result["score"]["tail"]
                rank = result["rank"]["tail"]
                new_score = new_result["score"]["tail"]
                new_rank = new_result["rank"]["tail"]

                print(dataset.printable_triple(pred))
                print(f"\tDirect score: from {str(score)} to {str(new_score)}")
                print(f"\tTail rank: from {str(rank)} to {str(new_rank)}")
                print()

                conversions.append(
                    {
                        "triples_to_add": [
                            dataset.labels_triple(triple)
                            for triple in triple_to_convert_to_added[pred]
                        ],
                        "score": str(score),
                        "rank": str(rank),
                        "new_score": str(new_score),
                        "new_rank": str(new_rank),
                    }
                )
            evaluation["conversions"] = conversions
            evaluations.append(evaluation)

    elif mode == "necessary":
        triple_to_best_rule = defaultdict(list)
        for explanation in explanations:
            pred = dataset.ids_triple(explanation["triple"])
            preds.append(pred)
            best_rule, _ = explanation["rule_to_relevance"][0]
            best_rule = [dataset.ids_triple(triple) for triple in best_rule]

            triple_to_best_rule[pred] = best_rule

        triples_to_remove = []

        for pred in preds:
            triples_to_remove += triple_to_best_rule[pred]

        new_dataset = copy.deepcopy(dataset)
        print("Removing triples: ")
        for s, p, o in triples_to_remove:
            print(f"\t{dataset.printable_triple((s, p, o))}")

        new_dataset.remove_training_triples(triples_to_remove)

        results = model.predict_triples(numpy.array(preds))
        results = {triple: result for triple, result in zip(preds, results)}
        new_model = model_class(dataset=new_dataset, hp=model_hp, init_random=True)

        hp = model_config["training"]
        optimizer_params = optimizer_class.get_hyperparams_class()(**hp)
        optimizer = optimizer_class(model=new_model, hp=optimizer_params)
        optimizer.train(training_triples=new_dataset.training_triples)
        new_model.eval()

        new_results = new_model.predict_triples(numpy.array(preds))
        new_results = {triple: result for triple, result in zip(preds, new_results)}

        evaluations = []
        for pred in preds:
            result = results[pred]
            new_result = new_results[pred]

            score = result["score"]["tail"]
            rank = result["rank"]["tail"]
            new_score = new_result["score"]["tail"]
            new_rank = new_result["rank"]["tail"]

            print(dataset.printable_triple(pred))
            print(f"\tDirect score: from {str(score)} to {str(new_score)}")
            print(f"\tTail rank: from {str(rank)} to {str(new_rank)}")
            print()

            evaluation = {
                "triple_to_explain": dataset.labels_triple(pred),
                "rule": [
                    dataset.labels_triple(triple)
                    for triple in triple_to_best_rule[pred]
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
    main()
