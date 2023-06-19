import pytest

import numpy as np

from src.tests.prefilters.fixtures import *

from src.prefilters import TypeBasedPreFilter
from src.tests.prefilters.type_based_prefilter.utils import (
    get_fake_triples,
    remove_related_triples,
)


@pytest.fixture
def fake_most_promising_triple(dataset):
    return dataset.ids_triple(
        (
            "/m/0bbm7r",
            "/award/award_winning_work/awards_won./award/award_honor/award",
            "/m/07z2lx",
        )
    )


@pytest.fixture
def fake_dataset(dataset, fake_most_promising_triple, pred):
    _, _, tail_to_explain = triple_to_explain
    (
        most_promising_head,
        most_promising_relation,
        _,
    ) = fake_most_promising_triple
    remove_related_triples(dataset, fake_most_promising_triple)

    fake_triples = get_fake_triples(
        dataset, tail_to_explain, most_promising_head, most_promising_relation
    )
    dataset.add_training_triples(fake_triples)

    return dataset


@pytest.fixture
def type_based_prefilter(fake_dataset):
    return TypeBasedPreFilter(fake_dataset)


def test_head_to_explain_as_tail(
    type_based_prefilter,
    pred,
    fake_most_promising_triple,
):
    [most_promising_triple] = type_based_prefilter.most_promising_triples_for(
        triple_to_explain=pred,
        perspective="head",
        top_k=1,
        verbose=True,
    )

    assert most_promising_triple == fake_most_promising_triple
