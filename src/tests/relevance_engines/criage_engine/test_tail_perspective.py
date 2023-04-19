import pytest
import numpy as np

from src.tests.relevance_engines.fixtures import *

from src.relevance_engines import CriageEngine


@pytest.fixture
def highest_relevance_sample(dataset):
    return dataset.fact_to_sample(
        (
            "/m/0fc9js",
            "/award/award_category/category_of",
            "/m/0gcf2r",
        )
    )


@pytest.fixture
def lowest_relevance_sample(dataset):
    return dataset.fact_to_sample(
        (
            "/m/04g2jz2",
            "/award/award_category/category_of",
            "/m/0gcf2r",
        )
    )


@pytest.fixture
def criage_engine(dataset, model, hyperparameters):
    return CriageEngine(
        model=model,
        dataset=dataset,
        hyperparameters=hyperparameters,
    )


def test_removal_relevance_tail_perspective(
    criage_engine,
    sample_to_explain,
    highest_relevance_sample,
    lowest_relevance_sample,
):
    hr_fake_samples_relevance = criage_engine.removal_relevance(
        sample_to_explain=sample_to_explain,
        samples_to_remove=np.array([highest_relevance_sample]),
        perspective="tail",
    )

    lr_fake_sample_relevance = criage_engine.removal_relevance(
        sample_to_explain=sample_to_explain,
        samples_to_remove=np.array([lowest_relevance_sample]),
        perspective="tail",
    )

    assert hr_fake_samples_relevance < lr_fake_sample_relevance


def test_addition_relevance_tail_perspective(
    criage_engine,
    sample_to_explain,
    highest_relevance_sample,
    lowest_relevance_sample,
):
    hr_fake_samples_relevance = criage_engine.addition_relevance(
        sample_to_convert=sample_to_explain,
        samples_to_add=np.array([highest_relevance_sample]),
        perspective="tail",
    )

    lr_fake_sample_relevance = criage_engine.addition_relevance(
        sample_to_convert=sample_to_explain,
        samples_to_add=np.array([lowest_relevance_sample]),
        perspective="tail",
    )

    assert hr_fake_samples_relevance > lr_fake_sample_relevance
