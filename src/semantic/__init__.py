from SPARQLWrapper import SPARQLWrapper, JSON

from .. import DB100K_PATH, DB100K_REASONED_PATH
from .. import DBPEDIA50_PATH, DBPEDIA50_REASONED_PATH
from .. import YAGO4_20_PATH, YAGO4_20_REASONED_PATH

from .semantic_similarity import (
    compute_semantic_similarity_entities,
    compute_semantic_similarity_triples,
)

sparql = SPARQLWrapper(endpoint="http://dbpedia.org/sparql")
sparql.setReturnFormat(JSON)
