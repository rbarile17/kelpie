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
        "dimension": 200,
        "input_dropout_rate": trial.suggest_categorical(
            "input_dropout_rate", [0, 0.1, 0.2]
        ),
        "hidden_dropout_rate": trial.suggest_categorical(
            "hidden_dropout_rate",
            [
                0,
                0.1,
                0.2,
            ],
        ),
        "feature_map_dropout_rate": trial.suggest_categorical(
            "feature_map_dropout_rate", [0, 0.1, 0.3, 0.5]
        ),
        "hidden_layer_size": 9728,
    }

    print(f"Initializing model {model}...")
    model_class = MODEL_REGISTRY[model]["class"]
    optimizer_class = MODEL_REGISTRY[model]["optimizer"]
    hyperparams_class = model_class.get_hyperparams_class()
    model_hp = hyperparams_class(**model_hp)
    model = model_class(dataset=dataset, hp=model_hp, init_random=True)
    model.to("cuda")

    optimizer_hp = {
        "batch_size": 512,
        "label_smoothing": 0.1,
        "epochs": 1000,
        "lr": trial.suggest_float("lr", 0, 0.1),
        "decay": 0.995,
    }

    optimizer_params = optimizer_class.get_hyperparams_class()(**optimizer_hp)
    optimizer = optimizer_class(model=model, hp=optimizer_params)



    training_triples = dataset.training_triples
    valid_triples = dataset.validation_triples

    num_samples = int(0.10 * training_triples.shape[0])
    sampled_indices = np.random.choice(training_triples.shape[0], num_samples, replace=False)
    sampled_training_triples = training_triples[sampled_indices]

    num_samples = int(0.10 * valid_triples.shape[0])
    sampled_indices = np.random.choice(valid_triples.shape[0], num_samples, replace=False)
    sampled_triples = valid_triples[sampled_indices]

    optimizer.train(sampled_training_triples, None, 5, sampled_triples, trial)

    evaluator = Evaluator(model=model)
    metrics = evaluator.evaluate(triples=sampled_triples)

    return metrics["h1"]


@click.command()
@click.option("--dataset", type=click.Choice(DATASETS))
def main(dataset):
    set_seeds(42)

    print(f"Loading dataset {dataset}...")
    dataset_name = dataset
    dataset = Dataset(dataset=dataset_name)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, "ConvE", dataset),
        n_trials=100,
        timeout=8 * 60 * 60,
    )

    print("Best trial:", study.best_trial.number)
    print("Best H@1:", study.best_trial.value)
    print("Best hyperparameters:", study.best_params)

    with open(f"params_{dataset_name}.json", "w") as f:
        json.dump(study.best_params, f)


if __name__ == "__main__":
    main()
