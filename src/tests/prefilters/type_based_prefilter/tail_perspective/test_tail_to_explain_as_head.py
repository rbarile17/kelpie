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
            "/m/0gcf2r",
            "/common/topic/webpage./common/webpage/category",
            "/m/08mbj5d",
        )
    )


@pytest.fixture
def fake_dataset(dataset, fake_most_promising_triple, triple_to_explain):
    head_to_explain, _, _ = triple_to_explain
    (
        _,
        most_promising_relation,
        most_promising_tail,
    ) = fake_most_promising_triple
    remove_related_triples(dataset, fake_most_promising_triple)

    fake_triples = get_fake_triples(
        dataset, head_to_explain, most_promising_tail, most_promising_relation, True
    )
    dataset.add_training_triples(fake_triples)

    return dataset


@pytest.fixture
def type_based_prefilter(fake_dataset):
    return TypeBasedPreFilter(None, fake_dataset)


def test_tail_to_explain_as_head(
    type_based_prefilter,
    triple_to_explain,
    fake_most_promising_triple,
):
    [most_promising_triple] = type_based_prefilter.most_promising_triples_for(
        triple_to_explain=triple_to_explain,
        perspective="tail",
        top_k=1,
        verbose=True,
    )

    assert most_promising_triple == fake_most_promising_triple
