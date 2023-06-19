import pytest

from src.tests.prefilters.fixtures import *
from src.prefilters import NoPreFilter


@pytest.fixture
def triples_featuring_head(dataset, pred):
    triples_featuring_head = []
    pred_head, _, _ = pred
    for head, relation, tail in dataset.training_triples:
        if head == pred_head or tail == pred_head:
            triples_featuring_head.append((head, relation, tail))

    return set(triples_featuring_head)


@pytest.fixture
def triples_featuring_tail(dataset, pred):
    triples_featuring_tail = []
    _, _, pred_tail = pred
    for head, relation, tail in dataset.training_triples:
        if head == pred_tail or tail == pred_tail:
            triples_featuring_tail.append((head, relation, tail))

    return set(triples_featuring_tail)


@pytest.fixture
def no_prefilter(dataset):
    return NoPreFilter(dataset)


def test_head_perspective(no_prefilter, pred, triples_featuring_head):
    most_promising_triples = no_prefilter.most_promising_triples_for(
        triple_to_explain=pred,
        perspective="head",
        top_k=1,
        verbose=True,
    )

    assert set(most_promising_triples) == triples_featuring_head


def test_tail_persepective(no_prefilter, pred, triples_featuring_tail):
    most_promising_triples = no_prefilter.most_promising_triples_for(
        triple_to_explain=pred,
        perspective="tail",
        top_k=1,
        verbose=True,
    )

    assert set(most_promising_triples) == triples_featuring_tail
