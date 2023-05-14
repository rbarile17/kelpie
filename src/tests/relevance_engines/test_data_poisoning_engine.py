import pytest
import numpy as np

from src.tests.relevance_engines.fixtures import *

from src.relevance_engines import DataPoisoningEngine
from src.link_prediction.models import LEARNING_RATE


@pytest.fixture
def data_poisoning_engine(dataset, model, hyperparameters):
    return DataPoisoningEngine(
        model=model,
        dataset=dataset,
        hyperparameters=hyperparameters,
        epsilon=hyperparameters[LEARNING_RATE],
    )


def test_removal_relevance(
    data_poisoning_engine,
    triple_to_explain,
    highest_relevance_triple,
    lowest_relevance_triple,
):
    hr_fake_triples_relevance = data_poisoning_engine.removal_relevance(
        triple_to_explain=triple_to_explain,
        triples_to_remove=np.array([highest_relevance_triple]),
        perspective="head",
    )[0]

    lr_fake_triple_relevance = data_poisoning_engine.removal_relevance(
        triple_to_explain=triple_to_explain,
        triples_to_remove=np.array([lowest_relevance_triple]),
        perspective="head",
    )[0]

    assert hr_fake_triples_relevance > lr_fake_triple_relevance


def test_addition_relevance(
    data_poisoning_engine,
    triple_to_explain,
    highest_relevance_triple,
    lowest_relevance_triple,
):
    hr_fake_triples_relevance = data_poisoning_engine.addition_relevance(
        triple_to_convert=triple_to_explain,
        triples_to_add=np.array([highest_relevance_triple]),
        perspective="head",
    )[0]

    lr_fake_triple_relevance = data_poisoning_engine.addition_relevance(
        triple_to_convert=triple_to_explain,
        triples_to_add=np.array([lowest_relevance_triple]),
        perspective="head",
    )[0]

    assert hr_fake_triples_relevance > lr_fake_triple_relevance
