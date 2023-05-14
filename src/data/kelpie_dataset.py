import copy

import numpy as np

from collections import defaultdict

from .dataset import Dataset


class KelpieDataset():
    def __init__(self, dataset: Dataset, entity_id):
        self.dataset = dataset
        self.to_filter = copy.deepcopy(dataset.to_filter)
        self.train_to_filter = copy.deepcopy(dataset.train_to_filter)
        self.entity_to_id = copy.deepcopy(dataset.entity_to_id)
        self.id_to_entity = copy.deepcopy(dataset.id_to_entity)
        self.relation_to_id = copy.deepcopy(dataset.relation_to_id)
        self.id_to_relation = copy.deepcopy(dataset.id_to_relation)

        self.num_entities = dataset.num_entities + 1
        self.num_relations = dataset.num_relations

        self.original_entity_id = entity_id
        self.original_entity_label = self.id_to_entity[entity_id]
        self.kelpie_entity_id = self.num_entities - 1
        self.kelpie_entity_label = "kelpie_" + self.original_entity_label
        self.entity_to_id[self.kelpie_entity_label] = self.kelpie_entity_id
        self.id_to_entity[self.kelpie_entity_id] = self.kelpie_entity_label

        self.kelpie_training_triples = Dataset.replace_entity_in_triples(
            dataset.entity_to_training_triples[self.original_entity_id],
            self.original_entity_id,
            self.kelpie_entity_id,
        )
        self.kelpie_validation_triples = Dataset.replace_entity_in_triples(
            dataset.entity_to_validation_triples[self.original_entity_id],
            self.original_entity_id,
            self.kelpie_entity_id,
        )
        self.kelpie_testing_triples = Dataset.replace_entity_in_triples(
            dataset.entity_to_testing_triples[self.original_entity_id],
            self.original_entity_id,
            self.kelpie_entity_id,
        )
        self.kelpie_training_triples_copy = copy.deepcopy(self.kelpie_training_triples)
        self.kelpie_validation_triples_copy = copy.deepcopy(
            self.kelpie_validation_triples
        )
        self.kelpie_testing_triples_copy = copy.deepcopy(self.kelpie_testing_triples)

        triples_to_stack = [self.kelpie_training_triples]
        if len(self.kelpie_validation_triples) > 0:
            triples_to_stack.append(self.kelpie_validation_triples)
        if len(self.kelpie_testing_triples) > 0:
            triples_to_stack.append(self.kelpie_testing_triples)
        all_kelpie_triples = np.vstack(triples_to_stack)
        for head, relation, tail in self.kelpie_training_triples:
            self.train_to_filter[(head, relation)].append(tail)
            self.train_to_filter[(tail, relation + self.num_relations)].append(head)
        for head, relation, tail in all_kelpie_triples:
            self.to_filter[(head, relation)].append(tail)
            self.to_filter[(tail, relation + self.num_relations)].append(head)

        self.kelpie_training_triple_to_index = {}
        for i in range(len(self.kelpie_training_triples)):
            current_head, current_relation, current_tail = self.kelpie_training_triples[
                i
            ]
            self.kelpie_training_triple_to_index[
                (current_head, current_relation, current_tail)
            ] = i

        self.last_added_triples = []
        self.last_added_triples_number = 0
        self.last_filter_additions = defaultdict(lambda: [])
        self.last_added_kelpie_triples = []

        self.last_removed_triples = []
        self.last_removed_triples_number = 0
        self.last_filter_removals = defaultdict(lambda: [])
        self.last_removed_kelpie_triples = []

    def as_kelpie_triple(self, original_triple):
        if not self.original_entity_id in original_triple:
            raise Exception(
                "Could not find the original entity "
                + str(self.original_entity_id)
                + " in the passed triple "
                + str(original_triple)
            )

        return Dataset.replace_entity_in_triple(
            triple=original_triple,
            old_entity=self.original_entity_id,
            new_entity=self.kelpie_entity_id,
        )

    def add_training_triples(self, triples_to_add: np.array):
        """
        Add a set of training triples to the training triples of the kelpie entity of this KelpieDataset.
        The triples to add must still feature the original entity id; this method will convert them before addition.
        The KelpieDataset will keep track of the last performed addition so it can be undone if necessary
        calling the undo_last_training_triples_addition method.

        :param triples_to_add: the triples to add, still featuring the id of the original entity,
                               in the form of a numpy array
        """

        for triple in triples_to_add:
            assert (
                self.original_entity_id == triple[0]
                or self.original_entity_id == triple[2]
            )

        self.last_added_triples = triples_to_add
        self.last_added_triples_number = len(triples_to_add)

        # reset all data structures needed to undo additions. We only want to keep track of the *last* addition.
        self.last_filter_additions = defaultdict(lambda: [])
        self.last_added_kelpie_triples = []

        kelpie_triples_to_add = Dataset.replace_entity_in_triples(
            triples_to_add,
            old_entity=self.original_entity_id,
            new_entity=self.kelpie_entity_id,
        )
        for cur_head, cur_rel, cur_tail in kelpie_triples_to_add:
            self.to_filter[(cur_head, cur_rel)].append(cur_tail)
            self.to_filter[(cur_tail, cur_rel + self.num_relations)].append(
                cur_head
            )
            self.train_to_filter[(cur_head, cur_rel)].append(cur_tail)
            self.train_to_filter[
                (cur_tail, cur_rel + self.num_relations)
            ].append(cur_head)

            self.last_added_kelpie_triples.append((cur_head, cur_rel, cur_tail))
            self.last_filter_additions[(cur_head, cur_rel)].append(cur_tail)
            self.last_filter_additions[
                (cur_tail, cur_rel + self.num_relations)
            ].append(cur_head)

        self.kelpie_training_triples = np.vstack(
            (self.kelpie_training_triples, np.array(kelpie_triples_to_add))
        )

    def remove_training_triples(self, triples_to_remove: np.array):
        for triple in triples_to_remove:
            assert (
                self.original_entity_id == triple[0]
                or self.original_entity_id == triple[2]
            )

        self.last_removed_triples = triples_to_remove
        self.last_removed_triples_number = len(triples_to_remove)

        # reset data structures needed to undo removals. We only want to keep track of the *last* removal.
        self.last_filter_removals = defaultdict(list)
        self.last_removed_kelpie_triples = []

        kelpie_training_triples_to_remove = Dataset.replace_entity_in_triples(
            triples=triples_to_remove,
            old_entity=self.original_entity_id,
            new_entity=self.kelpie_entity_id,
            as_numpy=False,
        )

        # update to_filter and train_to_filter
        for (
            current_head,
            current_relation,
            current_tail,
        ) in kelpie_training_triples_to_remove:
            self.to_filter[(current_head, current_relation)].remove(current_tail)
            self.to_filter[
                (current_tail, current_relation + self.num_relations)
            ].remove(current_head)
            self.train_to_filter[(current_head, current_relation)].remove(current_tail)
            self.train_to_filter[
                (current_tail, current_relation + self.num_relations)
            ].remove(current_head)

            # and also update the data structures required for undoing the removal
            self.last_removed_kelpie_triples.append(
                (current_head, current_relation, current_tail)
            )
            self.last_filter_removals[(current_head, current_relation)].append(
                current_tail
            )
            self.last_filter_removals[
                (current_tail, current_relation + self.num_relations)
            ].append(current_head)

        # get the indices of the triples to remove in the kelpie_training_triples structure
        # and use them to perform the actual removal
        kelpie_train_indices_to_remove = [
            self.kelpie_training_triple_to_index[x]
            for x in kelpie_training_triples_to_remove
        ]
        self.kelpie_training_triples = np.delete(
            self.kelpie_training_triples, kelpie_train_indices_to_remove, axis=0
        )

    def undo_last_training_triples_removal(self):
        """
        This method undoes the last removal performed on this KelpieDataset
        calling its add_training_triples method.

        The purpose of undoing the removals performed on a pre-existing KelpieDataset,
        instead of creating a new KelpieDataset from scratch, is to improve efficiency.
        """
        if self.last_removed_triples_number <= 0:
            raise Exception("No removal to undo.")

        # revert the self.kelpie_training_triples to the self.kelpie_training_triples_copy
        self.kelpie_training_triples = copy.deepcopy(self.kelpie_training_triples_copy)

        # undo additions to to_filter and train_to_filter
        for key in self.last_filter_removals:
            for x in self.last_filter_removals[key]:
                self.to_filter[key].append(x)
                self.train_to_filter[key].append(x)

        # reset the data structures used to undo additions
        self.last_removed_triples = []
        self.last_removed_triples_number = 0
        self.last_filter_removals = defaultdict(list)
        self.last_removed_kelpie_triples = []

    def undo_last_training_triples_addition(self):
        """
        This method undoes the last addition performed on this KelpieDataset
        calling its add_training_triples method.

        The purpose of undoing the additions performed on a pre-existing KelpieDataset,
        instead of creating a new KelpieDataset from scratch, is to improve efficiency.
        """

        if self.last_added_triples_number <= 0:
            raise Exception("No addition to undo.")

        # revert the self.kelpie_train_triples to the self.kelpie_train_triples_copy
        self.kelpie_training_triples = copy.deepcopy(self.kelpie_training_triples_copy)

        # undo additions to to_filter and train_to_filter
        for key in self.last_filter_additions:
            for x in self.last_filter_additions[key]:
                self.to_filter[key].remove(x)
                self.train_to_filter[key].remove(x)

        # reset the data structures used to undo additions
        self.last_added_triples = []
        self.last_added_triples_number = 0
        self.last_filter_additions = defaultdict(lambda: [])
        self.last_added_kelpie_triples = []

    def as_original_triple(self, kelpie_triple):
        if not self.kelpie_entity_id in kelpie_triple:
            raise Exception(
                "Could not find the original entity "
                + str(self.original_entity_id)
                + " in the passed sample "
                + str(kelpie_triple)
            )
        return Dataset.replace_entity_in_triple(
            sample=kelpie_triple,
            old_entity=self.kelpie_entity_id,
            new_entity=self.original_entity_id,
        )


    def invert_triples(self, triples):
        return self.dataset.invert_triples(triples)


    def printable_triple(self, triple):
        return self.dataset.printable_triple(triple)
