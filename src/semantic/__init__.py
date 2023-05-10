from SPARQLWrapper import SPARQLWrapper, JSON

from ..dataset import Dataset

dataset = Dataset(name="DBpedia50", separator="\t", load=True)

sparql = SPARQLWrapper(endpoint="http://dbpedia.org/sparql")
sparql.setReturnFormat(JSON)
