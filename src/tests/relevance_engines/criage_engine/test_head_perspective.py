import pytest
import numpy as np

from src.tests.relevance_engines.fixtures import *

from src.relevance_engines import CriageEngine


@pytest.fixture
def highest_relevance_triple(dataset):
    return dataset.ids_triple(
        (
            "/m/04m_zp",
            "/award/award_nominee/award_nominations./award/award_nomination/award",
            "/m/07z2lx",
        )
    )


@pytest.fixture
def lowest_relevance_triple(dataset):
    return dataset.ids_triple(
        (
            "/m/0bbm7r",
            "/award/award_winning_work/awards_won./award/award_honor/award",
            "/m/07z2lx",
        )
    )


@pytest.fixture
def criage_engine(dataset, model, hyperparameters):
    return CriageEngine(
        model=model,
        dataset=dataset,
        hyperparameters=hyperparameters,
    )


def test_removal_relevance_head_perspective(
    criage_engine,
    triple_to_explain,
    highest_relevance_triple,
    lowest_relevance_triple,
):
    hr_fake_triples_relevance = criage_engine.removal_relevance(
        triple_to_explain=triple_to_explain,
        triples_to_remove=np.array([highest_relevance_triple]),
        perspective="head",
    )

    lr_fake_triple_relevance = criage_engine.removal_relevance(
        triple_to_explain=triple_to_explain,
        triples_to_remove=np.array([lowest_relevance_triple]),
        perspective="head",
    )

    assert hr_fake_triples_relevance < lr_fake_triple_relevance


def test_addition_relevance_head_perspective(
    criage_engine,
    triple_to_explain,
    highest_relevance_triple,
    lowest_relevance_triple,
):
    hr_fake_triples_relevance = criage_engine.addition_relevance(
        triple_to_convert=triple_to_explain,
        triples_to_add=np.array([highest_relevance_triple]),
        perspective="head",
    )

    lr_fake_triple_relevance = criage_engine.addition_relevance(
        triple_to_convert=triple_to_explain,
        triples_to_add=np.array([lowest_relevance_triple]),
        perspective="head",
    )

    assert hr_fake_triples_relevance > lr_fake_triple_relevance
