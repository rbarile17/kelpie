from pathlib import Path
from SPARQLWrapper import SPARQLWrapper, JSON

from ..dataset import DATA_PATH, DBPEDIA50
from ..dataset import Dataset

DATA_PATH = Path(DATA_PATH)
DBPEDIA50_PATH = DATA_PATH / DBPEDIA50
DBPEDIA50_REASONED_PATH = DBPEDIA50_PATH / "reasoned"

dataset = Dataset(name="DBpedia50", separator="\t", load=True)

sparql = SPARQLWrapper(endpoint="http://dbpedia.org/sparql")
sparql.setReturnFormat(JSON)
