import pytest

from src.tests.fixtures import *


@pytest.fixture
def best_triple(dataset):
    return dataset.ids_triple(
        (
            "/m/07z2lx",
            "/award/award_category/winners./award/award_honor/ceremony",
            "/m/05c1t6z",
        )
    )


@pytest.fixture
def worst_triple(dataset):
    return dataset.ids_triple(
        (
            "/m/03cv_gy",
            "/award/award_winning_work/awards_won./award/award_honor/award",
            "/m/07z2lx",
        )
    )
