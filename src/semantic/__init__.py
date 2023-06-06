from SPARQLWrapper import SPARQLWrapper, JSON

from .. import DBPEDIA50_PATH, DBPEDIA50_REASONED_PATH

from ..data import Dataset
from .semantic_similarity import (
    compute_semantic_similarity_entities,
    compute_semantic_similarity_triples,
)

dataset = Dataset(dataset="DBpedia50")

sparql = SPARQLWrapper(endpoint="http://dbpedia.org/sparql")
sparql.setReturnFormat(JSON)
