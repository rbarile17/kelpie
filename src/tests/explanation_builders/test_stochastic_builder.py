from src.tests.explanation_builders.fixtures import *

from src.explanation_builders import StochasticBuilder
from src.relevance_engines import RelevanceEngine


class MockEngine(RelevanceEngine):
    def compute_relevance(self, pred, triples):
        map = {(1, 1, 1): 2, (2, 2, 2): 1}
        [triple] = triples
        return map[triple]


@pytest.fixture
def builder(dataset):
    engine = MockEngine(None, dataset)
    return StochasticBuilder(1, engine, max_explanation_length=1)


def test_acceptance_criteria(builder):
    pred = (None, None, None)
    candidate_triples = [(1, 1, 1), (2, 2, 2)]

    best_triple = builder.build_explanations(pred, candidate_triples)[0][0][0]

    assert best_triple == (1, 1, 1)
