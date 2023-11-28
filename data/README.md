<meta name="robots" content="noindex">

# Datasets

We perform the experiments on three datasets: DBpedia50k, DB100K and YAGO4-20. 

DB50K and DB100K are both samples of the DBpedia KG. 

We sampled YAGO4-20 from the YAGO4 KG. We extracted triples that mention entities involved in at least 20 triples. Next, we removed triples with literal objects.

The common aspect of these datasets is that along with the RDF triples they include or can be integrated with OWL (specifically OWL 2 DL) statements including the classes of the entities, as well as other schema axioms concerning classes and relationships. Such datasets can be regarded as ontologies with assertional (RDF) and terminological (OWL) knowledge. We require this feature as our approach heavily relies on a function _Cl_ returning the conjunction of the classes an entity belongs to. Moreover, the OWL schema axioms enable us to utilize the HermiT reasoner to enrich the output of _Cl_ with classes to which entities implicitly belong. We run the reasoner offline materializing all the implicit information. Hence, we execute our method without any other adjustment.

We highlight that for DB50k and DB100K only the RDF triples were available off the shelf. We retrieved the classes of the entities with custom SPARQL queries to the DBpedia endpoint, while the schema axioms are provided in the full version of DBpedia. Such schema axioms are defined in the `DBpedia.owl` file stored both in `data/DBpedia50` and in `data/DB100K`.

The integrated datasets are provided in the `data` folder. However, we describe the process to build them, focusing on DB100K. Initially the available files are `data/DB100K/train.txt`, `data/DB100K/valid.txt`, `data/DB100K/test.txt` and `data/DB100K/DBpedia.owl`.
The txt files with the triples are defined using the ID of the entities in Wikidata instead of DBpedia, hence we computed a mapping by running:
```python
python -m src.semantic.DB100K.retrieve_mapping
python -m src.semantic.DB100K.replace_mapped_entities
```
obtainig the files `data/DB100K/mapped/train.txt`, `data/DB100K/mapped/valid.txt`, `data/DB100K/test.txt` which contains RDF triples on entities with the ID in DBpedia.

Now we proceed to integrate the triples with the semantic information. We run the command:
```python
python -m src.semantic.retrieve_classes
```
to obtain `data/DB100K/entities.csv` containing the classes to which each entity in DB100K belongs.
Then we run
```python
python -m src.semantic.load_ontology
```
to integrate the triples and the retrieved classes with the owl ontology obtaining the file `data/DB100K/DB100K.owl`.
Next we run
```python
python -m src.semantic.reason
```
to materialize all the implicit information through the HermiT reasoner obtaining the file `data/DB100K/reasoned/DB100K_reasoned.owl`
Finally we run
```python
python -m src.semantic.reasoned_onto_get_classes
```
to obtain the file `data/DB100K/reasoned/entities.csv` which contains for each entity the classes to which it belongs, including the implicit ones. This file on which our extensions to Kelpie rely.

The process in analogous for DBpedia50.

While, for YAGO4-20 the process is a bit different. Firstly, 