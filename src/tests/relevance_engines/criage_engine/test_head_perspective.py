import pytest
import numpy as np

from src.tests.relevance_engines.fixtures import *

from src.relevance_engines import NecessaryCriageEngine, SufficientCriageEngine


@pytest.fixture
def best_triple(dataset):
    return dataset.ids_triple(
        (
            "/m/04m_zp",
            "/award/award_nominee/award_nominations./award/award_nomination/award",
            "/m/07z2lx",
        )
    )


@pytest.fixture
def worst_triple(dataset):
    return dataset.ids_triple(
        (
            "/m/0bbm7r",
            "/award/award_winning_work/awards_won./award/award_honor/award",
            "/m/07z2lx",
        )
    )


@pytest.fixture
def necessary_engine(dataset, model):
    return NecessaryCriageEngine(model, dataset)


@pytest.fixture
def sufficient_engine(dataset, model):
    return SufficientCriageEngine(model, dataset)


def test_necessary_relevance(necessary_engine, pred, best_triple, worst_triple):
    hr = necessary_engine.compute_relevance(pred, best_triple, "head")
    lr = necessary_engine.compute_relevance(pred, worst_triple, "head")

    assert hr < lr


def test_sufficient_relevance(sufficient_engine, pred, best_triple, worst_triple):
    sufficient_engine.select_entities_to_convert(pred, 10)
    hr = sufficient_engine.compute_relevance(pred, best_triple, "head")
    lr = sufficient_engine.compute_relevance(pred, worst_triple, "head")

    assert hr > lr
