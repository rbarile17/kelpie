from src.tests.explanation_builders.fixtures import *

from src.explanation_builders import StochasticSufficientExplanationBuilder


@pytest.fixture
def stochastic_necessary_builder(model, dataset, hyperparameters, sample_to_explain):
    return StochasticSufficientExplanationBuilder(
        model=model,
        dataset=dataset,
        hyperparameters=hyperparameters,
        sample_to_explain=sample_to_explain,
        perspective="head"
    )


def test_build_explanations(
    stochastic_necessary_builder, known_explanation_sample, known_not_in_explanation_sample
):
    explanation_sample = stochastic_necessary_builder.build_explanations(
        [known_explanation_sample, known_not_in_explanation_sample], top_k=1
    )[0][0][0]

    assert explanation_sample == known_explanation_sample
