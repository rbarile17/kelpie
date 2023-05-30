import torch
import numpy as np

from collections import defaultdict

from pykeen.datasets import get_dataset

from .names import ONE_TO_ONE, ONE_TO_MANY, MANY_TO_ONE, MANY_TO_MANY
from .. import DBPEDIA50_PATH


class Dataset:
    def __init__(self, dataset: str):
        self.dataset_name = dataset
        if dataset == "DBpedia50":
            self.dataset = get_dataset(
                training=DBPEDIA50_PATH / "train.txt",
                testing=DBPEDIA50_PATH / "test.txt",
                validation=DBPEDIA50_PATH / "valid.txt",
            )
        else:
            self.dataset = get_dataset(dataset=dataset)
        self.id_to_entity = {v: k for k, v in self.dataset.entity_to_id.items()}
        self.id_to_relation = {v: k for k, v in self.dataset.relation_to_id.items()}

        self.entity_to_training_triples = defaultdict(list)
        self.entity_to_validation_triples = defaultdict(list)
        self.entity_to_testing_triples = defaultdict(list)
        for head, relation, tail in self.training_triples:
            self.entity_to_training_triples[head].append((head, relation, tail))
            self.entity_to_training_triples[tail].append((head, relation, tail))
        for head, relation, tail in self.validation_triples:
            self.entity_to_validation_triples[head].append((head, relation, tail))
            self.entity_to_validation_triples[tail].append((head, relation, tail))
        for head, relation, tail in self.testing_triples:
            self.entity_to_testing_triples[head].append((head, relation, tail))
            self.entity_to_testing_triples[tail].append((head, relation, tail))

        for entity in self.entity_to_training_triples:
            self.entity_to_training_triples[entity] = list(
                set(self.entity_to_training_triples[entity])
            )
        for entity in self.entity_to_validation_triples:
            self.entity_to_training_triples[entity] = list(
                set(self.entity_to_training_triples[entity])
            )
        for entity in self.entity_to_testing_triples:
            self.entity_to_training_triples[entity] = list(
                set(self.entity_to_training_triples[entity])
            )

        self.entity_to_degree = {
            e: len(triples) for e, triples in self.entity_to_training_triples.items()
        }

        self.train_to_filter = defaultdict(list)
        for head, relation, tail in self.training_triples:
            self.train_to_filter[(head, relation)].append(tail)
            self.train_to_filter[(tail, relation + self.num_relations)].append(head)

        self.to_filter = defaultdict(list)
        for head, relation, tail in self.all_triples:
            self.to_filter[(head, relation)].append(tail)
            self.to_filter[(tail, relation + self.num_relations)].append(head)

        self._compute_relation_to_type()

    @property
    def training(self):
        return self.dataset.training

    @property
    def training_triples(self):
        return self.dataset.training.mapped_triples.numpy()

    @property
    def validation(self):
        return self.dataset.validation

    @property
    def validation_triples(self):
        return self.dataset.validation.mapped_triples.numpy()

    @property
    def testing(self):
        return self.dataset.testing

    @property
    def testing_triples(self):
        return self.dataset.testing.mapped_triples.numpy()

    @property
    def all_triples(self):
        return np.vstack(
            [self.training_triples, self.validation_triples, self.testing_triples]
        )

    @property
    def num_entities(self):
        return self.dataset.num_entities

    @property
    def num_relations(self):
        return self.dataset.num_relations

    @property
    def entity_to_id(self):
        return self.dataset.entity_to_id

    @property
    def relation_to_id(self):
        return self.dataset.relation_to_id

    def __iter__(self):
        return self.dataset.__iter__()

    def labels_triple(self, ids_triple):
        h, r, t = ids_triple
        return (self.id_to_entity[h], self.id_to_relation[r], self.id_to_entity[t])

    def labels_triples(self, ids_triples):
        return [self.labels_triple(ids_triple) for ids_triple in ids_triples]

    def ids_triple(self, labels_triple):
        h, r, t = labels_triple
        return (self.entity_to_id[h], self.relation_to_id[r], self.entity_to_id[t])

    def ids_triples(self, labels_triples):
        return [self.ids_triple(labels_triple) for labels_triple in labels_triples]

    def printable_triple(self, triple):
        h, r, t = self.labels_triple(triple)
        return f"<{h}, {r}, {t}>"

    def add_training_triple(self, triple):
        head, relation, tail = triple
        self.dataset.training.mapped_triples = torch.cat(
            [
                self.dataset.training.mapped_triples,
                torch.tensor([[head, relation, tail]]),
            ],
            dim=0,
        )
        self.entity_to_training_triples[head].append(triple)
        self.entity_to_training_triples[tail].append(triple)
        self.entity_to_degree[head] += 1
        self.entity_to_degree[tail] += 1
        self.to_filter[(head, relation)].append(tail)
        self.train_to_filter[(head, relation)].append(tail)

    def add_training_triples(self, triples):
        for triple in triples:
            self.add_training_triple(triple)

    def remove_training_triple(self, triple):
        head, relation, tail = triple
        self.dataset.training.mapped_triples = self.dataset.training.mapped_triples[
            ~(
                (self.dataset.training.mapped_triples[:, 0] == head)
                & (self.dataset.training.mapped_triples[:, 1] == relation)
                & (self.dataset.training.mapped_triples[:, 2] == tail)
            )
        ]
        self.entity_to_training_triples[head].remove(triple)
        if head != tail:
            self.entity_to_training_triples[tail].remove(triple)
        self.entity_to_degree[head] -= 1
        if head != tail:
            self.entity_to_degree[tail] -= 1
        self.to_filter[(head, relation)].remove(tail)
        self.train_to_filter[(head, relation)].remove(tail)

    def remove_training_triples(self, triples):
        for triple in set(triples):
            self.remove_training_triple(triple)

    def _compute_relation_to_type(self):
        """
        This method computes the type of each relation in the dataset based on the self.train_to_filter structure
        (that must have been already computed and populated).
        The mappings relation - relation type are written in the self.relation_to_type dict.
        :return: None
        """
        if len(self.train_to_filter) == 0:
            raise Exception(
                "The dataset has not been loaded yet, so it is not possible to compute relation types yet."
            )

        relation_to_heads_nums = defaultdict(list)
        relation_to_tails_nums = defaultdict(list)

        for entity, relation in self.train_to_filter:
            if relation >= self.num_relations:
                relation_to_heads_nums[relation - self.num_relations].append(
                    len(self.to_filter[(entity, relation)])
                )
            else:
                relation_to_tails_nums[relation].append(
                    len(self.to_filter[(entity, relation)])
                )

        self.relation_to_type = {}

        for relation in relation_to_heads_nums:
            average_heads_per_tail = np.average(relation_to_heads_nums[relation])
            average_tails_per_head = np.average(relation_to_tails_nums[relation])

            if average_heads_per_tail > 1.2 and average_tails_per_head > 1.2:
                self.relation_to_type[relation] = MANY_TO_MANY
            elif average_heads_per_tail > 1.2 and average_tails_per_head <= 1.2:
                self.relation_to_type[relation] = MANY_TO_ONE
            elif average_heads_per_tail <= 1.2 and average_tails_per_head > 1.2:
                self.relation_to_type[relation] = ONE_TO_MANY
            else:
                self.relation_to_type[relation] = ONE_TO_ONE

    def invert_triples(self, triples: np.array):
        """
        This method computes and returns the inverted version of the passed triples.
        :param triples: the direct triples to invert, in the form of a numpy array
        :return: the corresponding inverse triples, in the form of a numpy array
        """
        output = np.copy(triples)

        output[:, 0] = output[:, 2]
        output[:, 2] = triples[:, 0]
        output[:, 1] += self.num_relations

        return output

    @staticmethod
    def replace_entity_in_triple(triple, old_entity: int, new_entity: int):
        head, relation, tail = triple
        if head == old_entity:
            head = new_entity
        if tail == old_entity:
            tail = new_entity
        return (head, relation, tail)

    @staticmethod
    def replace_entity_in_triples(triples, old_entity: int, new_entity: int):
        results = []
        for head, relation, tail in triples:
            if head == old_entity:
                head = new_entity
            if tail == old_entity:
                tail = new_entity
            results.append((head, relation, tail))

        return results

    def printable_nple(self, nple: list):
        return " +\n\t\t".join([self.printable_triple(sample) for sample in nple])
