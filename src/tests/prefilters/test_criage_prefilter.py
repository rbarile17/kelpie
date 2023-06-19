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
def fake_dataset(dataset, pred, fake_most_promising_triple):
    pred_head, _, pred_tail = pred
    triples_to_remove = []
    for head, relation, tail in dataset.training_triples:
        if tail == pred_head or tail == pred_tail:
            triples_to_remove.append((head, relation, tail))
    dataset.remove_training_triples(triples_to_remove)
    dataset.add_training_triple(fake_most_promising_triple)

    return dataset


@pytest.fixture
def criage_prefilter(fake_dataset):
    return CriagePreFilter(fake_dataset)


def test_criage_prefilter(criage_prefilter, pred, fake_most_promising_triple):
    [most_promising_triple] = criage_prefilter.most_promising_triples_for(
        pred=pred,
        perspective="",
        top_k=1,
        verbose=True,
    )

    assert most_promising_triple == fake_most_promising_triple
