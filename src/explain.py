import json
import click

import torch

from . import BASELINES, DATASETS, MODELS_PATH
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
    help="Path of the the predictions to explain.",
)
@click.option(
    "--coverage",
    type=int,
    default=10,
    help="Number of entities to convert (sufficient mode only).",
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
):
    set_seeds(42)

    model_config = json.load(open(model_config, "r"))
    model = model_config["model"]
    model_path = model_config.get("model_path", MODELS_PATH / f"{model}_{dataset}.pt")

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

    explanations = []
    for pred in preds:
        s, p, o = pred
        print(f"\nExplaining pred: <{s}, {p}, {o}>")
        pred = dataset.ids_triple(pred)
        explanation = pipeline.explain(pred=pred, prefilter_k=prefilter_threshold)

        explanations.append(explanation)

    with open("output.json", "w") as output:
        json.dump(explanations, output)


if __name__ == "__main__":
    main()
