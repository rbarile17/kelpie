import pytest

import numpy as np

from src.tests.prefilters.fixtures import *
from src.prefilters import CriagePreFilter


@pytest.fixture
def fake_most_promising_sample(dataset):
    return dataset.fact_to_sample(
        (
            "/m/07z2lx",
            "/award/award_category/category_of",
            "/m/0gcf2r",
        )
    )


@pytest.fixture
def fake_dataset(dataset, sample_to_explain, fake_most_promising_sample):
    head_to_explain, _, tail_to_explain = sample_to_explain
    samples_to_remove = []
    for head, relation, tail in dataset.train_samples:
        if tail == head_to_explain or tail == tail_to_explain:
            samples_to_remove.append((head, relation, tail))
    dataset.remove_training_samples(np.array(samples_to_remove))
    dataset.add_training_samples(np.array([fake_most_promising_sample]))

    return dataset


@pytest.fixture
def criage_prefilter(fake_dataset):
    return CriagePreFilter(None, fake_dataset)


def test_criage_prefilter(
    criage_prefilter, sample_to_explain, fake_most_promising_sample
):
    [most_promising_sample] = criage_prefilter.most_promising_samples_for(
        sample_to_explain=sample_to_explain,
        perspective="",
        top_k=1,
        verbose=True,
    )

    assert most_promising_sample == fake_most_promising_sample
