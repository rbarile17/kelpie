import itertools


pairs = lambda iterable: itertools.combinations(iterable, 2)


def jaccard_similarity(set1, set2):
    if len(set1) == 0 or len(set2) == 0:
        return 0
    return len(set1.intersection(set2)) / len(set1.union(set2))
