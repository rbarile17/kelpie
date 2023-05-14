from src.tests.explanation_builders.fixtures import *

from src.explanation_builders import CriageNecessaryExplanationBuilder


@pytest.fixture
def known_explanation_triple(dataset):
    return dataset.ids_triple(
        (
            "/m/04m_zp",
            "/award/award_nominee/award_nominations./award/award_nomination/award",
            "/m/07z2lx",
        )
    )


@pytest.fixture
def known_not_in_explanation_triple(dataset):
    return dataset.ids_triple(
        (
            "/m/0bbm7r",
            "/award/award_winning_work/awards_won./award/award_honor/award",
            "/m/07z2lx",
        )
    )


@pytest.fixture
def criage_necessary_builder(model, dataset, hyperparameters, triple_to_explain):
    return CriageNecessaryExplanationBuilder(
        model=model,
        dataset=dataset,
        hyperparameters=hyperparameters,
        triple_to_explain=triple_to_explain,
        perspective="",
    )


def test_build_explanations(
    criage_necessary_builder, known_explanation_triple, known_not_in_explanation_triple
):
    explanation_triple = criage_necessary_builder.build_explanations(
        [known_explanation_triple, known_not_in_explanation_triple], top_k=1
    )[0][0][0]

    assert explanation_triple == known_explanation_triple
