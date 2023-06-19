import itertools
import random

import numpy as np

from .explanation_builder import ExplanationBuilder

from .. import key


class StochasticBuilder(ExplanationBuilder):
    def __init__(self, xsi, engine, max_explanation_length: int = 4):
        dataset = engine.dataset
        super().__init__(dataset=dataset, max_explanation_length=max_explanation_length)

        self.window_size = 10

        self.xsi = xsi
        self.engine = engine

    def build_explanations(self, pred, candidate_triples: list, k: int = 10):
        triple_to_rel = self.explore_singleton_rules(pred, candidate_triples)

        triple_to_rel_sorted = triple_to_rel.items()
        triple_to_rel_sorted = sorted(triple_to_rel_sorted, key=key, reverse=True)
        rule_to_rel = [((t,), rel) for (t, rel) in triple_to_rel_sorted]

        triples_number = len(triple_to_rel)
        rels_num = triples_number
        _, best = rule_to_rel[0]
        explore_compound_rules = True
        if best > self.xsi:
            explore_compound_rules = False

        if explore_compound_rules:
            for rule_length in range(2, min(triples_number, self.length_cap) + 1):
                (cur_rule_to_rel, cur_rels_num) = self.explore_compound_rules(
                    pred, candidate_triples, rule_length, triple_to_rel
                )
                rels_num += cur_rels_num
                cur_rule_to_rel = cur_rule_to_rel.items()
                cur_rule_to_rel = sorted(cur_rule_to_rel, key=key, reverse=True)
                rule_to_rel += cur_rule_to_rel

                _, current_best = cur_rule_to_rel[0]
                if current_best > best:
                    best = current_best
                if best > self.xsi:
                    break

                rule_length += 1

        rule_to_rel = sorted(rule_to_rel, key=key, reverse=True)
        relevances = [relevance for (_, relevance) in rule_to_rel]
        variance = np.var(np.array(relevances))
        rule_to_rel = rule_to_rel[:k]

        return rule_to_rel

    def explore_singleton_rules(self, pred, triples: list):
        triple_to_relevance = {}

        for triple in triples:
            printable_triple = self.dataset.printable_nple(triple)
            print(f"\tComputing relevance for rule: {printable_triple}")

            relevance = self.engine.compute_relevance(pred, triple)
            triple_to_relevance[triple] = relevance
            print(f"\tRelevance = {relevance:.3f}")
        return triple_to_relevance

    def explore_compound_rules(
        self, pred, triples: list, length: int, triple_to_relevance: dict
    ):
        rules = itertools.combinations(triples, length)
        rules = [(r, self.compute_rule_prescore(r, triple_to_relevance)) for r in rules]
        rules = sorted(rules, key=lambda x: x[1], reverse=True)

        terminate = False
        best = -1e6
        sliding_window = [None for _ in range(self.window_size)]

        rule_to_relevance = {}
        computed_relevances = 0
        for i, (rule, _) in enumerate(rules):
            if terminate:
                break

            print(
                "\tComputing relevance for rule: \n\t\t"
                f"{self.dataset.printable_nple(rule)}"
            )
            relevance = self.engine.compute_relevance(pred, rule)
            print(f"\tRelevance = {relevance:.3f}")
            rule_to_relevance[rule] = relevance
            computed_relevances += 1

            sliding_window[i % self.window_size] = relevance

            if relevance > self.xsi:
                return rule_to_relevance, computed_relevances
            elif relevance >= best:
                best = relevance
            elif i >= self.window_size:
                avg_window_relevance = sum(sliding_window) / self.window_size
                terminate_threshold = avg_window_relevance / best
                random_value = random.random()
                terminate = random_value > terminate_threshold

                print()
                print(f"\tRelevance {relevance:.3f}")
                print(f"\tAverage window relevance {avg_window_relevance:.3f}")
                print(f"\tMax relevance seen so far {best:.3f}")
                print(f"\tTerminate threshold: {terminate_threshold:.3f}")
                print(f"\tRandom value: {random_value:.3f}")
                print(f"\tTerminate: {str(terminate)}")

        return rule_to_relevance, computed_relevances

    def compute_rule_prescore(self, rule, triple_to_relevance):
        # semantic_similarity = compute_semantic_similarity_triples(
        #     self.dataset,
        #     rule,
        #     self.head_to_explain,
        # )

        # relevances = [triple_to_relevance[x] for x in rule]
        # relevances = MinMaxScaler().fit_transform(np.array(relevances).reshape(-1, 1))
        # relevances = relevances.reshape(-1)

        # relevance = sum(relevances) / len(rule)

        # return semantic_similarity + relevance

        return sum([triple_to_relevance[triple] for triple in rule])
