import pytest

from src.tests.relevance_engines.fixtures import *

from src.relevance_engines import (
    NecessaryPostTrainingEngine,
    SufficientPostTrainingEngine,
)


@pytest.fixture
def necessary_engine(dataset, model, training_params):
    engine = NecessaryPostTrainingEngine(model, dataset, training_params)
    engine.set_cache()

    return engine


@pytest.fixture
def sufficient_engine(dataset, model, training_params):
    engine = SufficientPostTrainingEngine(model, dataset, training_params)
    engine.set_cache()

    return engine


def test_necessary_relevance(necessary_engine, pred, best_triple, worst_triple):
    hr = necessary_engine.compute_relevance(pred, [best_triple])
    lr = necessary_engine.compute_relevance(pred, [worst_triple])

    assert hr > lr


def test_sufficient_relevance(sufficient_engine, pred, best_triple, worst_triple):
    sufficient_engine.select_entities_to_convert(pred, 10)
    hr = sufficient_engine.compute_relevance(pred, [best_triple])
    lr = sufficient_engine.compute_relevance(pred, [worst_triple])

    assert hr > lr
