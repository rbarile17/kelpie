import pytest
import numpy as np

from src.tests.relevance_engines.fixtures import *

from src.relevance_engines import CriageEngine


@pytest.fixture
def highest_relevance_sample(dataset):
    return dataset.fact_to_sample(
        (
            "/m/04m_zp",
            "/award/award_nominee/award_nominations./award/award_nomination/award",
            "/m/07z2lx",
        )
    )


@pytest.fixture
def lowest_relevance_sample(dataset):
    return dataset.fact_to_sample(
        (
            "/m/0bbm7r",
            "/award/award_winning_work/awards_won./award/award_honor/award",
            "/m/07z2lx",
        )
    )


@pytest.fixture
def criage_engine(dataset, model, hyperparameters):
    return CriageEngine(
        model=model,
        dataset=dataset,
        hyperparameters=hyperparameters,
    )


def test_removal_relevance_head_perspective(
    criage_engine,
    sample_to_explain,
    highest_relevance_sample,
    lowest_relevance_sample,
):
    hr_fake_samples_relevance = criage_engine.removal_relevance(
        sample_to_explain=sample_to_explain,
        samples_to_remove=np.array([highest_relevance_sample]),
        perspective="head",
    )

    lr_fake_sample_relevance = criage_engine.removal_relevance(
        sample_to_explain=sample_to_explain,
        samples_to_remove=np.array([lowest_relevance_sample]),
        perspective="head",
    )

    assert hr_fake_samples_relevance < lr_fake_sample_relevance


def test_addition_relevance_head_perspective(
    criage_engine,
    sample_to_explain,
    highest_relevance_sample,
    lowest_relevance_sample,
):
    hr_fake_samples_relevance = criage_engine.addition_relevance(
        sample_to_convert=sample_to_explain,
        samples_to_add=np.array([highest_relevance_sample]),
        perspective="head",
    )

    lr_fake_sample_relevance = criage_engine.addition_relevance(
        sample_to_convert=sample_to_explain,
        samples_to_add=np.array([lowest_relevance_sample]),
        perspective="head",
    )

    assert hr_fake_samples_relevance > lr_fake_sample_relevance
