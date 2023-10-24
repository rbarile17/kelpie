import click
import json
import optuna

import numpy as np

from . import MODEL_REGISTRY

from .. import DATASETS

from .evaluation import Evaluator

from ..data import Dataset
from ..utils import set_seeds


def objective(trial, model, dataset):
    model_hp = {
        "norm": trial.suggest_int("norm", 1, 2),
        "dimension": 2 ** trial.suggest_int("dimension", 6, 8),
    }

    print(f"Initializing model {model}...")
    model_class = MODEL_REGISTRY[model]["class"]
    optimizer_class = MODEL_REGISTRY[model]["optimizer"]
    hyperparams_class = model_class.get_hyperparams_class()
    model_hp = hyperparams_class(**model_hp)
    model = model_class(dataset=dataset, hp=model_hp, init_random=True)
    model.to("cuda")

    optimizer_hp = {
        "batch_size": 2048,
        "epochs": 1000,
        "lr": trial.suggest_float("lr", 0, 0.1),
        "margin": trial.suggest_categorical("margin", [1, 2, 5, 10]),
        "negative_triples_ratio": trial.suggest_categorical(
            "negative_triples_ratio", [5, 10, 15]
        ),
        "regularizer_weight": 0
    }

    optimizer_params = optimizer_class.get_hyperparams_class()(**optimizer_hp)
    optimizer = optimizer_class(model=model, hp=optimizer_params)

    training_triples = dataset.training_triples
    valid_triples = dataset.validation_triples

    num_samples = int(0.10 * valid_triples.shape[0])
    sampled_indices = np.random.choice(valid_triples.shape[0], num_samples, replace=False)
    sampled_triples = valid_triples[sampled_indices]

    optimizer.train(training_triples, None, 5, sampled_triples, trial)

    evaluator = Evaluator(model=model)
    metrics = evaluator.evaluate(triples=sampled_triples)

    return metrics["h1"]


@click.command()
@click.option("--dataset", type=click.Choice(DATASETS))
def main(dataset):
    set_seeds(42)

    print(f"Loading dataset {dataset}...")
    dataset = Dataset(dataset=dataset)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, "TransE", dataset),
        n_trials=100,
    )

    print("Best trial:", study.best_trial.number)
    print("Best H@1:", study.best_trial.value)
    print("Best hyperparameters:", study.best_params)

    with open("params.json", "w") as f:
        json.dump(study.best_params, f)


if __name__ == "__main__":
    main()
