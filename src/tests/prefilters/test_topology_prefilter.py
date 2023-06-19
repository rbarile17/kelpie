import pytest

from src.tests.prefilters.fixtures import *

from src.prefilters import TopologyPreFilter


@pytest.fixture
def fake_most_promising_triple(dataset):
    return dataset.ids_triple(
        ("/m/07z2lx", "/award/award_category/disciplines_or_subjects", "/m/0gcf2r")
    )


@pytest.fixture
def fake_dataset(dataset, fake_most_promising_triple):
    dataset.add_training_triple(fake_most_promising_triple)

    return dataset


@pytest.fixture
def topology_prefilter(fake_dataset):
    return TopologyPreFilter(fake_dataset)


def test_head_perspective(
    topology_prefilter, pred, fake_most_promising_triple
):
    [most_promising_triple] = topology_prefilter.most_promising_triples_for(
        triple_to_explain=pred,
        perspective="head",
        top_k=1,
        verbose=True,
    )

    assert most_promising_triple == fake_most_promising_triple
