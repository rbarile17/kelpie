import pandas as pd

from tqdm import tqdm
from ast import literal_eval

from owlready2 import *

from . import DBPEDIA50_PATH
from src.dataset import Dataset


tqdm.pandas()
onto_path.append(str(DBPEDIA50_PATH))
onto = get_ontology("DBpedia.owl")
onto = onto.load()

dbr = onto.get_namespace("http://dbpedia.org/resource/")

entities = pd.read_csv(
    DBPEDIA50_PATH / "entities.csv",
    header=None,
    names=["entity", "classes"],
    converters={"classes": literal_eval},
)


def load_triple(triple):
    head, relation, tail = triple
    if head in entity_names and tail in entity_names:
        with dbr:
            value = dbr[head].__dict__.get(relation, [])
            value.append(dbr[tail])
            dbr[head].__class__(head, **{relation: value})


def check_disjoint_classes(classes):
    if len(classes) < 2:
        return False
    onto_classes = []
    for class_ in classes:
        try:
            onto_classes.append(onto[class_])
        except TypeError:
            pass
    onto_classes = [class_ for class_ in onto_classes if class_ is not None]
    classes_and_ancestors = []
    for class_ in onto_classes:
        classes_and_ancestors.append(class_)
        classes_and_ancestors.extend(class_.ancestors())
    for class_ in onto_classes:
        for disjoint_class in class_.disjoints():
            other_class = (
                disjoint_class.entities[0]
                if disjoint_class.entities[0] != class_
                else disjoint_class.entities[1]
            )

            if other_class in classes_and_ancestors:
                return True

        for ancestor in class_.ancestors():
            if ancestor.name != "Thing":
                for disjoint_class in ancestor.disjoints():
                    other_class = (
                        disjoint_class.entities[0]
                        if disjoint_class.entities[0] != ancestor
                        else disjoint_class.entities[1]
                    )
                    if other_class in classes_and_ancestors:
                        return True

    return False


entities["disjoint"] = entities["classes"].progress_map(check_disjoint_classes)
entities = entities[entities["disjoint"] == False]
entities = entities.drop(columns=["disjoint"])
entity_names = entities["entity"].tolist()
for _, (entity, classes) in entities.iterrows():
    instantiated_classes = 0
    for class_ in classes:
        try:
            with dbr:
                onto[class_](entity)
            instantiated_classes += 1
        except TypeError:
            # skip if class is not in ontology
            pass

    if instantiated_classes == 0:
        with dbr:
            Thing(entity)

dataset = Dataset(name="DBpedia50", separator="\t", load=True)

for triple in tqdm(dataset.train_triples):
    load_triple(triple)

for triple in tqdm(dataset.valid_triples):
    load_triple(triple)

for triple in tqdm(dataset.test_triples):
    load_triple(triple)

onto.save(str((DBPEDIA50_PATH / "DBpedia50.owl")))
