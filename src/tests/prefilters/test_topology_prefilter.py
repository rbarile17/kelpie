import numpy as np

import pytest

from src.tests.prefilters.fixtures import *

from src.prefilters import TopologyPreFilter


@pytest.fixture
def fake_most_promising_sample(dataset):
    return dataset.fact_to_sample(
        ("/m/07z2lx", "/award/award_category/disciplines_or_subjects", "/m/0gcf2r")
    )


@pytest.fixture
def fake_dataset(dataset, fake_most_promising_sample):
    dataset.add_training_samples(np.array([fake_most_promising_sample]))

    return dataset


@pytest.fixture
def topology_prefilter(fake_dataset):
    return TopologyPreFilter(None, fake_dataset)


def test_head_perspective(
    topology_prefilter, sample_to_explain, fake_most_promising_sample
):
    [most_promising_sample] = topology_prefilter.most_promising_samples_for(
        sample_to_explain=sample_to_explain,
        perspective="head",
        top_k=1,
        verbose=True,
    )

    assert most_promising_sample == fake_most_promising_sample


def test_tail_perspective(
    topology_prefilter, sample_to_explain, fake_most_promising_sample
):
    [most_promising_sample] = topology_prefilter.most_promising_samples_for(
        sample_to_explain=sample_to_explain,
        perspective="head",
        top_k=1,
        verbose=True,
    )

    assert most_promising_sample == fake_most_promising_sample
