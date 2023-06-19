from .relevance_engines import PostTrainingEngine


class Pipeline:
    def __init__(self, dataset, prefilter, builder):
        self.dataset = dataset
        self.prefilter = prefilter
        self.builder = builder

        self.engine = self.builder.engine
        self.model = self.engine.model

    def explain(self, pred, prefilter_k=-1):
        if isinstance(self.engine, PostTrainingEngine):
            self.engine.set_cache()
        print("\tRunning prefilter...")
        return self.prefilter.select_triples(pred=pred, k=prefilter_k)


class NecessaryPipeline(Pipeline):
    def __init__(self, dataset, prefilter, builder):
        super().__init__(dataset, prefilter, builder)

    def explain(self, pred, prefilter_k=-1):
        filtered_triples = super().explain(pred=pred, prefilter_k=prefilter_k)

        rule_to_relevance = self.builder.build_explanations(pred, filtered_triples)
        rule_to_relevance = [
            (self.dataset.labels_triples(rule), relevance)
            for rule, relevance in rule_to_relevance
        ]
        return {
            "triple": self.dataset.labels_triple(pred),
            "rule_to_relevance": rule_to_relevance,
        }


class SufficientPipeline(Pipeline):
    def __init__(self, dataset, prefilter, builder):
        super().__init__(dataset, prefilter, builder)

    def explain(self, pred, prefilter_k=50, to_convert_k=10):
        filtered_triples = super().explain(pred, prefilter_k)

        self.engine.select_entities_to_convert(
            pred=pred, k=to_convert_k, degree_cap=200
        )

        rule_to_relevance = self.builder.build_explanations(pred, filtered_triples)
        entities_to_conv = self.engine.entities_to_convert
        entities_to_conv = [self.dataset.id_to_entity[x] for x in entities_to_conv]
        rule_to_relevance = [
            (self.dataset.labels_triples(rule), relevance)
            for rule, relevance in rule_to_relevance
        ]
        return {
            "triple": self.dataset.labels_triple(pred),
            "entities_to_convert": entities_to_conv,
            "rule_to_relevance": rule_to_relevance,
        }
