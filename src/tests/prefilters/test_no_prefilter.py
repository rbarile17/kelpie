import pytest

from src.tests.prefilters.fixtures import *
from src.prefilters import NoPreFilter


@pytest.fixture
def triples_featuring_head(dataset, triple_to_explain):
    triples_featuring_head = []
    head_to_explain, _, _ = triple_to_explain
    for head, relation, tail in dataset.training_triples:
        if head == head_to_explain or tail == head_to_explain:
            triples_featuring_head.append((head, relation, tail))

    return set(triples_featuring_head)


@pytest.fixture
def triples_featuring_tail(dataset, triple_to_explain):
    triples_featuring_tail = []
    _, _, tail_to_explain = triple_to_explain
    for head, relation, tail in dataset.training_triples:
        if head == tail_to_explain or tail == tail_to_explain:
            triples_featuring_tail.append((head, relation, tail))

    return set(triples_featuring_tail)


@pytest.fixture
def no_prefilter(dataset):
    return NoPreFilter(None, dataset)


def test_head_perspective(
    no_prefilter, triple_to_explain, triples_featuring_head
):
    most_promising_triples = no_prefilter.most_promising_triples_for(
        triple_to_explain=triple_to_explain,
        perspective="head",
        top_k=1,
        verbose=True,
    )

    assert set(most_promising_triples) == triples_featuring_head


def test_tail_persepective(
    no_prefilter, triple_to_explain, triples_featuring_tail
):
    most_promising_triples = no_prefilter.most_promising_triples_for(
        triple_to_explain=triple_to_explain,
        perspective="tail",
        top_k=1,
        verbose=True,
    )

    assert set(most_promising_triples) == triples_featuring_tail
