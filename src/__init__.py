from pathlib import Path
from typing import Any, Tuple

MODELS_PATH = "models"
MODELS_PATH = Path(MODELS_PATH)
MAX_PROCESSES = 8

RESULTS_PATH = "results"
RESULTS_PATH = Path(RESULTS_PATH)

DATA_PATH = "data"
FB15K = "FB15k"
FB15K_237 = "FB15k-237"
WN18 = "WN18"
WN18RR = "WN18RR"
YAGO3_10 = "YAGO3-10"
DBPEDIA50 = "DBpedia50"
DB100K = "DB100K"

CONVE = "ConvE"
COMPLEX = "ComplEx"
TRANSE = "TransE"

DATASETS = [FB15K, FB15K_237, WN18, WN18RR, YAGO3_10, DBPEDIA50, DB100K]
BASELINES = ["k1", "data_poisoning", "criage"]

DATA_PATH = Path(DATA_PATH)
DBPEDIA50_PATH = DATA_PATH / DBPEDIA50
DBPEDIA50_REASONED_PATH = DBPEDIA50_PATH / "reasoned"

DB100K_PATH = DATA_PATH / DB100K
DB100K_MAPPED_PATH = DATA_PATH / DB100K / "mapped"
DB100K_REASONED_PATH = DB100K_PATH / "reasoned"

key = lambda x: x[1]

Triple = Tuple[Any, Any, Any]
