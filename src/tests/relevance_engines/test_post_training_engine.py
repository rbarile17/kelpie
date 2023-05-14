import pytest
import numpy as np

from src.tests.relevance_engines.fixtures import *

from src.relevance_engines import PostTrainingEngine


@pytest.fixture
def post_training_engine(dataset, model, hyperparameters):
    return PostTrainingEngine(
        model=model, dataset=dataset, hyperparameters=hyperparameters
    )


def test_removal_relevance(
    post_training_engine,
    triple_to_explain,
    highest_relevance_triple,
    lowest_relevance_triple,
):
    hr_fake_triples_relevance = post_training_engine.removal_relevance(
        triple_to_explain=triple_to_explain,
        triples_to_remove=np.array([highest_relevance_triple]),
        perspective="head",
    )[0]

    lr_fake_triple_relevance = post_training_engine.removal_relevance(
        triple_to_explain=triple_to_explain,
        triples_to_remove=np.array([lowest_relevance_triple]),
        perspective="head",
    )[0]

    assert hr_fake_triples_relevance > lr_fake_triple_relevance


def test_addition_relevance(
    post_training_engine,
    triple_to_explain,
    highest_relevance_triple,
    lowest_relevance_triple,
):
    hr_fake_triples_relevance = post_training_engine.addition_relevance(
        triple_to_convert=triple_to_explain,
        triples_to_add=np.array([highest_relevance_triple]),
        perspective="head",
    )[0]

    lr_fake_triple_relevance = post_training_engine.addition_relevance(
        triple_to_convert=triple_to_explain,
        triples_to_add=np.array([lowest_relevance_triple]),
        perspective="head",
    )[0]

    assert hr_fake_triples_relevance > lr_fake_triple_relevance
