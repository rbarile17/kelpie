from src.tests.explanation_builders.fixtures import *

from src.explanation_builders import CriageSufficientExplanationBuilder


@pytest.fixture
def known_explanation_triple_tail_perspective(dataset):
    return dataset.ids_triple(
        (
            "/m/02_3zj",
            "/award/award_category/category_of",
            "/m/0gcf2r",
        )
    )


@pytest.fixture
def known_not_in_explanation_triple_tail_perspective(dataset):
    return dataset.ids_triple(
        (
            "/m/07y9ts",
            "/time/event/instance_of_recurring_event",
            "/m/0gcf2r",
        )
    )


@pytest.fixture
def known_explanation_triple_head_perspective(dataset):
    return dataset.ids_triple(
        (
            "/m/04m_zp",
            "/award/award_nominee/award_nominations./award/award_nomination/award",
            "/m/07z2lx",
        )
    )


@pytest.fixture
def known_not_in_explanation_triple_head_perspective(dataset):
    return dataset.ids_triple(
        (
            "/m/05m9f9",
            "/award/award_nominee/award_nominations./award/award_nomination/award",
            "/m/07z2lx",
        )
    )


@pytest.fixture
def criage_sufficient_builder(model, dataset, hyperparameters, triple_to_explain):
    criage_sufficient_builder = CriageSufficientExplanationBuilder(
        model=model,
        dataset=dataset,
        hyperparameters=hyperparameters,
        triple_to_explain=triple_to_explain,
        perspective="head",
    )

    criage_sufficient_builder.entities_to_convert = [
        dataset.entity_to_id[entity]
        for entity in [
            "/m/03cffvv",
            "/m/01r216",
            "/m/02r22gf",
            "/m/01m7mv",
            "/m/0n_2q",
            "/m/0gct_",
            "/m/06929s",
            "/m/053yx",
            "/m/03kfl",
            "/m/011k1h",
        ]
    ]

    return criage_sufficient_builder


def test_head_perspective(
    criage_sufficient_builder,
    known_explanation_triple_head_perspective,
    known_not_in_explanation_triple_head_perspective,
):
    explanation_triple = criage_sufficient_builder.build_explanations(
        [
            known_explanation_triple_head_perspective,
            known_not_in_explanation_triple_head_perspective,
        ],
        top_k=1,
    )[0][0][0]

    assert explanation_triple == known_explanation_triple_head_perspective


def test_tail_perspective(
    criage_sufficient_builder,
    known_explanation_triple_tail_perspective,
    known_not_in_explanation_triple_tail_perspective,
):
    explanation_triple = criage_sufficient_builder.build_explanations(
        [
            known_explanation_triple_tail_perspective,
            known_not_in_explanation_triple_tail_perspective,
        ],
        top_k=1,
    )[0][0][0]

    assert explanation_triple == known_explanation_triple_tail_perspective
