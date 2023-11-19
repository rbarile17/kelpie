import html

import numpy as np
import pandas as pd

from tqdm import tqdm

from .models import Model, TransE


class Evaluator:
    def __init__(self, model: Model):
        self.model = model
        self.dataset = model.dataset

    def evaluate(self, triples: np.array, write_output: bool = False):
        self.model.cuda()

        batch_size = 256
        if len(triples) > batch_size:
            batch_start = 0
            results = []
            num_triples = len(triples)
            with tqdm(total=num_triples, unit="ex", leave=False) as p:
                while batch_start < num_triples:
                    batch_end = min(len(triples), batch_start + batch_size)
                    cur_batch = triples[batch_start:batch_end]
                    results += self.model.predict_triples(cur_batch)
                    batch_start += batch_size
                    p.update(batch_size)
        else:
            results = self.model.predict_triples(triples)

        ranks = [result["rank"] for result in results]
        if write_output:
            self.write_output(triples, ranks)

        all_ranks = []
        for i in range(triples.shape[0]):
            all_ranks.append(results[i]["rank"]["tail"])
            all_ranks.append(results[i]["rank"]["head"])

        return {
            "mrr": self.mrr(all_ranks),
            "h1": self.hits_at(all_ranks, 1),
            "h10": self.hits_at(all_ranks, 10),
            "mr": self.mr(all_ranks),
        }

    def write_output(self, triples, ranks):
        lines = []
        for i in range(triples.shape[0]):
            s, p, o = triples[i]

            head_rank, tail_rank = ranks[i]["head"], ranks[i]["tail"]

            s = self.dataset.id_to_entity[s]
            p = self.dataset.id_to_relation[p]
            o = self.dataset.id_to_entity[o]

            lines.append(
                {
                    "head": html.unescape(s),
                    "relation": html.unescape(p),
                    "tail": html.unescape(o),
                    "head_rank": head_rank,
                    "tail_rank": tail_rank,
                }
            )

        df = pd.DataFrame(lines)
        df.to_csv("ranks.csv", index=False, sep=";")

    @staticmethod
    def mrr(values):
        mrr = 0.0
        for value in values:
            mrr += 1.0 / float(value)
        mrr = mrr / float(len(values))
        return mrr

    @staticmethod
    def mr(values):
        return np.average(values)

    @staticmethod
    def hits_at(values, k: int):
        hits = 0
        for value in values:
            if value <= k:
                hits += 1
        return float(hits) / float(len(values))
