from pathlib import Path
from typing import Any, Tuple

MODELS_PATH = Path("models")
MAX_PROCESSES = 8

DATA_PATH = "data"
FB15K = "FB15k"
FB15K_237 = "FB15k-237"
WN18 = "WN18"
WN18RR = "WN18RR"
YAGO3_10 = "YAGO3-10"
DBPEDIA50 = "DBpedia50"
CONVE = "ConvE"
COMPLEX = "ComplEx"
TRANSE = "TransE"

DATASETS = [FB15K, FB15K_237, WN18, WN18RR, YAGO3_10, DBPEDIA50]
BASELINES = ["k1", "data_poisoning", "criage"]

DATA_PATH = Path(DATA_PATH)
DBPEDIA50_PATH = DATA_PATH / DBPEDIA50
DBPEDIA50_REASONED_PATH = DBPEDIA50_PATH / "reasoned"

key = lambda x: x[1]

Triple = Tuple[Any, Any, Any]
