import pytest

from src.tests.fixtures import *


@pytest.fixture
def highest_relevance_sample(dataset):
    return dataset.fact_to_sample(
        (
            "/m/07z2lx",
            "/award/award_category/winners./award/award_honor/ceremony",
            "/m/05c1t6z",
        )
    )


@pytest.fixture
def lowest_relevance_sample(dataset):
    return dataset.fact_to_sample(
        (
            "/m/03cv_gy",
            "/award/award_winning_work/awards_won./award/award_honor/award",
            "/m/07z2lx",
        )
    )
