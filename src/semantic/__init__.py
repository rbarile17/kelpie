from SPARQLWrapper import SPARQLWrapper, JSON

from .. import DBPEDIA50_PATH, DBPEDIA50_REASONED_PATH

from ..data import Dataset

dataset = Dataset(dataset="DBpedia50")

sparql = SPARQLWrapper(endpoint="http://dbpedia.org/sparql")
sparql.setReturnFormat(JSON)
