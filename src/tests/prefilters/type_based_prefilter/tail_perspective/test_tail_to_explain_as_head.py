import pytest

import numpy as np

from src.tests.prefilters.fixtures import *

from src.prefilters import TypeBasedPreFilter
from src.tests.prefilters.type_based_prefilter.utils import (
    get_fake_samples,
    remove_related_samples,
)


@pytest.fixture
def fake_most_promising_sample(dataset):
    return dataset.fact_to_sample(
        (
            "/m/0gcf2r",
            "/common/topic/webpage./common/webpage/category",
            "/m/08mbj5d",
        )
    )


@pytest.fixture
def fake_dataset(dataset, fake_most_promising_sample, sample_to_explain):
    head_to_explain, _, _ = sample_to_explain
    (
        _,
        most_promising_relation,
        most_promising_tail,
    ) = fake_most_promising_sample
    remove_related_samples(dataset, fake_most_promising_sample)

    fake_samples = get_fake_samples(
        dataset, head_to_explain, most_promising_tail, most_promising_relation, True
    )
    dataset.add_training_samples(fake_samples)

    return dataset


@pytest.fixture
def type_based_prefilter(fake_dataset):
    return TypeBasedPreFilter(None, fake_dataset)


def test_tail_to_explain_as_head(
    type_based_prefilter,
    sample_to_explain,
    fake_most_promising_sample,
):
    [most_promising_sample] = type_based_prefilter.most_promising_samples_for(
        sample_to_explain=sample_to_explain,
        perspective="tail",
        top_k=1,
        verbose=True,
    )

    assert most_promising_sample == fake_most_promising_sample
