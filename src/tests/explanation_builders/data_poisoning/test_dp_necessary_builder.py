import pytest

from src.tests.explanation_builders.fixtures import *

from src.explanation_builders import DataPoisoningNecessaryExplanationBuilder


@pytest.fixture
def dp_necessary_builder(model, dataset, hyperparameters, triple_to_explain):
    return DataPoisoningNecessaryExplanationBuilder(
        model=model,
        dataset=dataset,
        hyperparameters=hyperparameters,
        triple_to_explain=triple_to_explain,
        perspective="head",
    )


def test_build_explanations(
    dp_necessary_builder, known_explanation_triple, known_not_in_explanation_triple
):
    explanation_triple = dp_necessary_builder.build_explanations(
        [known_explanation_triple, known_not_in_explanation_triple], top_k=1
    )[0][0][0]

    assert explanation_triple == known_explanation_triple
