import pytest
import numpy as np

from src.tests.relevance_engines.fixtures import *

from src.relevance_engines import CriageEngine


@pytest.fixture
def highest_relevance_triple(dataset):
    return dataset.ids_triple(
        (
            "/m/0fc9js",
            "/award/award_category/category_of",
            "/m/0gcf2r",
        )
    )


@pytest.fixture
def lowest_relevance_triple(dataset):
    return dataset.ids_triple(
        (
            "/m/04g2jz2",
            "/award/award_category/category_of",
            "/m/0gcf2r",
        )
    )


@pytest.fixture
def criage_engine(dataset, model, hyperparameters):
    return CriageEngine(
        model=model,
        dataset=dataset,
        hyperparameters=hyperparameters,
    )


def test_removal_relevance_tail_perspective(
    criage_engine,
    triple_to_explain,
    highest_relevance_triple,
    lowest_relevance_triple,
):
    hr_fake_triples_relevance = criage_engine.removal_relevance(
        triple_to_explain=triple_to_explain,
        triples_to_remove=np.array([highest_relevance_triple]),
        perspective="tail",
    )

    lr_fake_triple_relevance = criage_engine.removal_relevance(
        triple_to_explain=triple_to_explain,
        triples_to_remove=np.array([lowest_relevance_triple]),
        perspective="tail",
    )

    assert hr_fake_triples_relevance < lr_fake_triple_relevance


def test_addition_relevance_tail_perspective(
    criage_engine,
    triple_to_explain,
    highest_relevance_triple,
    lowest_relevance_triple,
):
    hr_fake_triples_relevance = criage_engine.addition_relevance(
        triple_to_convert=triple_to_explain,
        triples_to_add=np.array([highest_relevance_triple]),
        perspective="tail",
    )

    lr_fake_triple_relevance = criage_engine.addition_relevance(
        triple_to_convert=triple_to_explain,
        triples_to_add=np.array([lowest_relevance_triple]),
        perspective="tail",
    )

    assert hr_fake_triples_relevance > lr_fake_triple_relevance
