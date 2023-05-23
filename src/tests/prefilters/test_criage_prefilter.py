import pytest

from src.tests.prefilters.fixtures import *
from src.prefilters import CriagePreFilter


@pytest.fixture
def fake_most_promising_triple(dataset):
    return dataset.ids_triple(
        (
            "/m/07z2lx",
            "/award/award_category/category_of",
            "/m/0gcf2r",
        )
    )


@pytest.fixture
def fake_dataset(dataset, triple_to_explain, fake_most_promising_triple):
    head_to_explain, _, tail_to_explain = triple_to_explain
    triples_to_remove = []
    for head, relation, tail in dataset.training_triples:
        if tail == head_to_explain or tail == tail_to_explain:
            triples_to_remove.append((head, relation, tail))
    dataset.remove_training_triples(triples_to_remove)
    dataset.add_training_triple(fake_most_promising_triple)

    return dataset


@pytest.fixture
def criage_prefilter(fake_dataset):
    return CriagePreFilter(fake_dataset)


def test_criage_prefilter(
    criage_prefilter, triple_to_explain, fake_most_promising_triple
):
    [most_promising_triple] = criage_prefilter.most_promising_triples_for(
        triple_to_explain=triple_to_explain,
        perspective="",
        top_k=1,
        verbose=True,
    )

    assert most_promising_triple == fake_most_promising_triple
