import click
import json
import os

import torch

from pathlib import Path

from . import BASELINES, DATASETS
from . import MODELS_PATH, RESULTS_PATH
from .link_prediction import MODEL_REGISTRY
from .prefilters import (
    TOPOLOGY_PREFILTER,
    TYPE_PREFILTER,
    NO_PREFILTER,
    WEIGHTED_TOPOLOGY_PREFILTER,
)

from .data import Dataset
from .explanation_builders import CriageBuilder, DataPoisoningBuilder, StochasticBuilder
from .explanation_builders.summarization import SUMMARIZATIONS
from .pipeline import NecessaryPipeline, SufficientPipeline
from .prefilters import (
    CriagePreFilter,
    NoPreFilter,
    TopologyPreFilter,
    TypeBasedPreFilter,
    WeightedTopologyPreFilter,
)
from .relevance_engines import (
    NecessaryCriageEngine,
    SufficientCriageEngine,
    NecessaryDPEngine,
    SufficientDPEngine,
    NecessaryPostTrainingEngine,
    SufficientPostTrainingEngine,
)
from .utils import set_seeds

PREFILTERS = [
    TOPOLOGY_PREFILTER,
    TYPE_PREFILTER,
    NO_PREFILTER,
    WEIGHTED_TOPOLOGY_PREFILTER,
]
modes = ["necessary", "sufficient"]


def build_pipeline(model, dataset, hp, mode, baseline, prefilter, xsi, summarization):
    prefilter_map = {
        TOPOLOGY_PREFILTER: TopologyPreFilter,
        TYPE_PREFILTER: TypeBasedPreFilter,
        NO_PREFILTER: NoPreFilter,
        WEIGHTED_TOPOLOGY_PREFILTER: WeightedTopologyPreFilter,
    }

    if mode == "necessary":
        if baseline == "criage":
            prefilter = CriagePreFilter(dataset)
            engine = NecessaryCriageEngine(model, dataset)
            builder = CriageBuilder(dataset, engine)
        elif baseline == "data_poisoning":
            prefilter = prefilter_map.get(prefilter, NoPreFilter)(dataset=dataset)
            engine = NecessaryDPEngine(model, dataset, hp["lr"])
            builder = DataPoisoningBuilder(dataset, engine)
        else:
            DEFAULT_XSI_THRESHOLD = 5
            xsi = xsi if xsi is not None else DEFAULT_XSI_THRESHOLD
            prefilter = prefilter_map.get(prefilter, TopologyPreFilter)(dataset=dataset)
            engine = NecessaryPostTrainingEngine(model, dataset, hp)
            builder = StochasticBuilder(xsi, engine, summarization=summarization)
        pipeline = NecessaryPipeline(dataset, prefilter, builder)
    elif mode == "sufficient":
        if baseline == "criage":
            prefilter = CriagePreFilter(dataset)
            engine = SufficientCriageEngine(model, dataset)
            builder = CriageBuilder(dataset, engine)
        elif baseline == "data_poisoning":
            engine = SufficientDPEngine(model, dataset)
            builder = DataPoisoningBuilder(dataset, engine)
        else:
            DEFAULT_XSI_THRESHOLD = 0.9
            xsi = xsi if xsi is not None else DEFAULT_XSI_THRESHOLD
            prefilter = prefilter_map.get(prefilter, TopologyPreFilter)(dataset=dataset)
            engine = SufficientPostTrainingEngine(model, dataset, hp)
            builder = StochasticBuilder(xsi, engine, summarization=summarization)
        pipeline = SufficientPipeline(dataset, prefilter, builder)

    return pipeline


@click.command()
@click.option("--dataset", type=click.Choice(DATASETS))
@click.option(
    "--model_config",
    type=click.Path(exists=True),
    help="Path of the model config (.json or .yml).",
)
@click.option(
    "--preds",
    type=click.Path(exists=True),
    help="Path of the predictions to explain.",
)
@click.option(
    "--coverage",
    type=int,
    default=10,
    help="Number of entities to convert (sufficient mode only).",
)
@click.option(
    "--skip",
    type=int,
    default=-1,
    help="Number of predictions to skip.",
)
@click.option("--baseline", type=click.Choice(BASELINES))
@click.option("--mode", type=click.Choice(modes))
@click.option(
    "--relevance_threshold",
    type=float,
    help="The relevance acceptance threshold.",
)
@click.option("--prefilter", type=click.Choice(PREFILTERS))
@click.option("--summarization", type=click.Choice(SUMMARIZATIONS))
@click.option(
    "--prefilter_threshold",
    type=int,
    default=20,
    help=f"The number of triples to select in pre-filtering.",
)
def main(
    dataset,
    model_config,
    preds,
    coverage,
    baseline,
    mode,
    prefilter,
    relevance_threshold,
    prefilter_threshold,
    summarization,
    skip
):
    set_seeds(42)

    model_config = json.load(open(model_config, "r"))
    model = model_config["model"]
    model_path = model_config.get("model_path", MODELS_PATH / f"{model}_{dataset}.pt")

    prefilter_short_names = {
        TOPOLOGY_PREFILTER: "bfs",
        TYPE_PREFILTER: "type",
        WEIGHTED_TOPOLOGY_PREFILTER: "wbfs",
    }
    prefilter_short_name = prefilter_short_names[prefilter] if prefilter else "bfs"
    summarization = summarization if summarization else "no"
    output_dir = f"{model}_{dataset}_{mode}_{prefilter_short_name}_th{prefilter_threshold}_{summarization}"

    print("Reading preds...")
    if preds is None:
        preds = f"preds/{model}_{dataset}.csv"
    with open(preds, "r") as preds:
        preds = [x.strip().split("\t") for x in preds.readlines()]

    print(f"Loading dataset {dataset}...")
    dataset = Dataset(dataset=dataset)

    print(f"Loading model {model}...")
    model_class = MODEL_REGISTRY[model]["class"]
    hyperparams_class = model_class.get_hyperparams_class()
    model_hp = hyperparams_class(**model_config["model_params"])
    model = model_class(dataset=dataset, hp=model_hp, init_random=True)
    model.to("cuda")

    model.load_state_dict(torch.load(model_path))
    model.eval()

    pipeline = build_pipeline(
        model,
        dataset,
        model_config["training"],
        mode,
        baseline,
        prefilter,
        relevance_threshold,
        summarization,
    )

    Path(RESULTS_PATH / output_dir).mkdir(exist_ok=True)

    explanations = []
    for i, pred in enumerate(preds):
        if i <= skip:
            continue
        s, p, o = pred
        print(f"\nExplaining pred {i}: <{s}, {p}, {o}>")
        pred = dataset.ids_triple(pred)
        explanation = pipeline.explain(pred=pred, prefilter_k=prefilter_threshold)

        explanations.append(explanation)

        with open(RESULTS_PATH / output_dir / "output.json", "w") as output:
            json.dump(explanations, output)

if __name__ == "__main__":
    main()
