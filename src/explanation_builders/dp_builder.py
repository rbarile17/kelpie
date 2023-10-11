from .explanation_builder import ExplanationBuilder
from .. import key


class DataPoisoningBuilder(ExplanationBuilder):
    def __init__(self, engine):
        super().__init__(engine.dataset, 1)
        self.engine = engine

    def build_explanations(self, pred, candidate_triples: list, k: int = 10):
        pred_s, _, _ = pred
        rule_to_relevance = {}

        for triple in candidate_triples:
            _, _, o = triple
            printable_triple = self.dataset.printable_triple(triple)
            print(f"\tComputing relevance for rule: {printable_triple}")

            perspective = "head" if o == pred_s else "tail"
            relevance = self.engine.compute_rule_relevance(triple, perspective)

            rule_to_relevance[triple] = relevance

        return sorted(rule_to_relevance.items(), key=key)[:k]