import time

import pandas as pd

from tqdm import tqdm

from . import dataset, sparql

tqdm.pandas()


def get_classes(entity):
    entity_uri = f"http://dbpedia.org/resource/{entity}"

    sparql.setQuery(
        f""" 
            SELECT DISTINCT ?class
            WHERE {{
                <{entity_uri}> a ?class .
                FILTER strstarts(str(?class), "http://dbpedia.org/ontology/")
                FILTER NOT EXISTS {{
                    ?subClass rdfs:subClassOf ?class .
                    <{entity_uri}> a ?subClass .
                    FILTER (?subClass != ?class)
                }}
            }}
            """
    )

    # wait a bit to avoid being blocked by DBpedia
    time.sleep(0.1)

    rows = sparql.queryAndConvert()["results"]["bindings"]
    return set([row["class"]["value"].split("/")[-1] for row in rows])


entities = pd.DataFrame(list(dataset.entity_to_id.keys()), columns=["entity"])
entities["classes"] = entities["entity"].progress_map(get_classes)

entities.to_csv("entities.csv", index=False, header=False)
