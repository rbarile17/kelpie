from src.tests.fixtures import *


@pytest.fixture
def known_explanation_sample(dataset):
    return dataset.fact_to_sample(
        (
            "/m/07z2lx",
            "/award/award_category/winners./award/award_honor/ceremony",
            "/m/05c1t6z",
        )
    )


@pytest.fixture
def known_not_in_explanation_sample(dataset):
    return dataset.fact_to_sample(
        (
            "/m/0415svh",
            "/award/award_nominee/award_nominations./award/award_nomination/award",
            "/m/07z2lx",
        )
    )