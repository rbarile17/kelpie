import pytest

from src.tests.prefilters.fixtures import *
from src.prefilters import NoPreFilter


@pytest.fixture
def samples_featuring_head(dataset, sample_to_explain):
    samples_featuring_head = []
    head_to_explain, _, _ = sample_to_explain
    for head, relation, tail in dataset.train_samples:
        if head == head_to_explain or tail == head_to_explain:
            samples_featuring_head.append((head, relation, tail))

    return samples_featuring_head


@pytest.fixture
def samples_featuring_tail(dataset, sample_to_explain):
    samples_featuring_tail = []
    _, _, tail_to_explain = sample_to_explain
    for head, relation, tail in dataset.train_samples:
        if head == tail_to_explain or tail == tail_to_explain:
            samples_featuring_tail.append((head, relation, tail))

    return samples_featuring_tail


@pytest.fixture
def no_prefilter(dataset):
    return NoPreFilter(None, dataset)


def test_head_perspective(
    no_prefilter, sample_to_explain, samples_featuring_head
):
    most_promising_samples = no_prefilter.most_promising_samples_for(
        sample_to_explain=sample_to_explain,
        perspective="head",
        top_k=1,
        verbose=True,
    )

    assert most_promising_samples == samples_featuring_head


def test_tail_persepective(
    no_prefilter, sample_to_explain, samples_featuring_tail
):
    most_promising_samples = no_prefilter.most_promising_samples_for(
        sample_to_explain=sample_to_explain,
        perspective="tail",
        top_k=1,
        verbose=True,
    )

    assert most_promising_samples == samples_featuring_tail
