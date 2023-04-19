import pytest
import numpy as np

from src.tests.relevance_engines.fixtures import *

from src.relevance_engines import PostTrainingEngine


@pytest.fixture
def post_training_engine(dataset, model, hyperparameters):
    return PostTrainingEngine(
        model=model, dataset=dataset, hyperparameters=hyperparameters
    )


def test_removal_relevance(
    post_training_engine,
    sample_to_explain,
    highest_relevance_sample,
    lowest_relevance_sample,
):
    hr_fake_samples_relevance = post_training_engine.removal_relevance(
        sample_to_explain=sample_to_explain,
        samples_to_remove=np.array([highest_relevance_sample]),
        perspective="head",
    )[0]

    lr_fake_sample_relevance = post_training_engine.removal_relevance(
        sample_to_explain=sample_to_explain,
        samples_to_remove=np.array([lowest_relevance_sample]),
        perspective="head",
    )[0]

    assert hr_fake_samples_relevance > lr_fake_sample_relevance


def test_addition_relevance(
    post_training_engine,
    sample_to_explain,
    highest_relevance_sample,
    lowest_relevance_sample,
):
    hr_fake_samples_relevance = post_training_engine.addition_relevance(
        sample_to_convert=sample_to_explain,
        samples_to_add=np.array([highest_relevance_sample]),
        perspective="head",
    )[0]

    lr_fake_sample_relevance = post_training_engine.addition_relevance(
        sample_to_convert=sample_to_explain,
        samples_to_add=np.array([lowest_relevance_sample]),
        perspective="head",
    )[0]

    assert hr_fake_samples_relevance > lr_fake_sample_relevance
