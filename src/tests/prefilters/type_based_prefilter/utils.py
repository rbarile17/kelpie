from collections import defaultdict


def get_fake_triples(
    dataset, entity_in_prediction, entity_to_analyze, relation_to_analyze, inverse=False
):
    direct_relations = defaultdict(int)
    inverse_relations = defaultdict(int)
    for head, relation, tail in dataset.training_triples:
        if head == entity_in_prediction:
            direct_relations[relation] += 1

        if tail == entity_in_prediction:
            inverse_relations[relation] += 1
    if inverse:
        inverse_relations[relation_to_analyze] -= 1
    else:
        direct_relations[relation_to_analyze] -= 1

    return [
        item
        for sublist in [
            [(entity_to_analyze, relation, 0) for _ in range(frequency)]
            for relation, frequency in direct_relations.items()
        ]
        for item in sublist
    ] + [
        item
        for sublist in [
            [(0, relation, entity_to_analyze) for _ in range(frequency)]
            for relation, frequency in inverse_relations.items()
        ]
        for item in sublist
    ]


def remove_related_triples(dataset, triple_to_analyze):
    head_to_analyze, _, tail_to_analyze = triple_to_analyze
    triples_to_remove = []
    for head, relation, tail in dataset.training_triples:
        if head == tail_to_analyze or head == head_to_analyze:
            triples_to_remove.append((head, relation, tail))
        if tail == tail_to_analyze or tail == head_to_analyze:
            triples_to_remove.append((head, relation, tail))
    dataset.remove_training_triples(set(triples_to_remove))
    dataset.add_training_triple(triple_to_analyze)
    