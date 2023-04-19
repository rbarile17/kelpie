from src.tests.explanation_builders.fixtures import *

from src.explanation_builders import CriageNecessaryExplanationBuilder


@pytest.fixture
def known_explanation_sample(dataset):
    return dataset.fact_to_sample(
        (
            "/m/04m_zp",
            "/award/award_nominee/award_nominations./award/award_nomination/award",
            "/m/07z2lx",
        )
    )


@pytest.fixture
def known_not_in_explanation_sample(dataset):
    return dataset.fact_to_sample(
        (
            "/m/0bbm7r",
            "/award/award_winning_work/awards_won./award/award_honor/award",
            "/m/07z2lx",
        )
    )


@pytest.fixture
def criage_necessary_builder(model, dataset, hyperparameters, sample_to_explain):
    return CriageNecessaryExplanationBuilder(
        model=model,
        dataset=dataset,
        hyperparameters=hyperparameters,
        sample_to_explain=sample_to_explain,
        perspective="",
    )


def test_build_explanations(
    criage_necessary_builder, known_explanation_sample, known_not_in_explanation_sample
):
    explanation_sample = criage_necessary_builder.build_explanations(
        [known_explanation_sample, known_not_in_explanation_sample], top_k=1
    )[0][0][0]

    assert explanation_sample == known_explanation_sample
