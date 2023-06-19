import click
import json

import torch

from .. import DATASETS, MODELS_PATH

from . import MODEL_REGISTRY
from .evaluation import Evaluator

from ..data import Dataset
from ..utils import set_seeds


@click.command()
@click.option("--dataset", type=click.Choice(DATASETS))
@click.option(
    "--model_config",
    type=click.Path(exists=True),
    help="Path of the model config (.json or .yml).",
)
def main(dataset, model_config):
    set_seeds(42)

    model_config = json.load(open(model_config, "r"))
    model = model_config["model"]
    model_path = model_config.get("model_path", MODELS_PATH / f"{model}_{dataset}.pt")

    print(f"Loading dataset {dataset}...")
    dataset = Dataset(dataset=dataset)

    print(f"Initializing model {model}...")
    model_class = MODEL_REGISTRY[model]["class"]
    hyperparams_class = model_class.get_hyperparams_class()
    model_hp = hyperparams_class(**model_config["model_params"])
    model = model_class(dataset=dataset, hp=model_hp, init_random=True)
    model.to("cuda")

    model.load_state_dict(torch.load(model_path))
    model.eval()

    print("Evaluating model...")
    evaluator = Evaluator(model=model)
    metrics = evaluator.evaluate(triples=dataset.testing_triples, write_output=True)
    print(f"Hits@1: {metrics['h1']:.3f}")
    print(f"Hits@10: {metrics['h10']:.3f}")
    print(f"Mean Reciprocal Rank: {metrics['mrr']:.3f}")
    print(f"Mean Rank: {metrics['mr']:.3f}")


if __name__ == "__main__":
    main()
