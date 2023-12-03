<meta name="robots" content="noindex">

# Kelpie

Kelpie is a post-hoc local explainability tool for Link Prediction (LP) on Knowledge Graphs (KGs) through embedding-based models.

Kelpie provides a simple and effective interface to identify the most relevant facts to any prediction; intuitively, when explaining a _object_ prediction <_s_, _p_, _o_>, Kelpie identifies the smallest set of training facts mentioning _s_ that are instrumental to that prediction.

Given a _object_ prediction <_s_, _p_, _o_>, Kelpie supports two explanation scenarios:
* a **necessary explanation** triples without which the model can’t make the prediction
* a **sufficient explanation** triples able to replicate the prediction on other entities

Kelpie is structured in a simple architecture based on the interaction of three modules. 

* **Pre-Filter**: selects the most promising triples featuring _s_
* **Explanation Builder**: combines the pre-filtered triples into candidate explanations and identifies valid ones according to relevance
* **Relevance Engine**: estimates the *relevance* of a candidate explanation

We provide two extensions of Kelpie. We incorporate in the Pre-Filter a measure of similarity between entities grounded on semantics. Moreover, we add a summarization step in the Explanation Builder and we adapt the algorithm to support such summarization. Specifically, there are three possible summarizations.

## Environment

We have run all our experiments on an environment configured as follows:
- OS: Pop!_OS 22.04 LTS
- Python 3.11.3
- CUDA Version: 11.7
- Driver Version: 535.113.01

Furthermore, this version of Kelpie requires the following libraries: 
- PyKEEN 1.10.1
- BisPy 0.2.2
- Pydantic 2.4.2

After cloning this repository (`git clone https://github.com/rbarile17/kelpie.git`) all the dependencies can be installed by running 
```bash
pip install -r requirements.txt
```

## Resources
We make the following resources available:

* the datasets in the `kelpie/data` folder
* the downloadable [trained models](https://figshare.com/s/83a1022644a5183244d8) to place in a new folder `kelpie/models`
* the output files obtained by running the experiments reported in our paper

## Usage
### Model Traning and Testing
Alternatively to download our trained models, it is possible to train the models from scratch.
We report the template of the command to execute the script `src/link_prediction/explain.py`
```python
python -m src.link_prediction.train --dataset <dataset> --model_config <config> --valid <validation_epochs>
```

### Explanation
We report how to extract explanations for a trained model, and how to verify their effectiveness. 

Each experiment is composed of two separate steps:
* **explanation extraction**: `src/explain.py`
* **explanation verification**: `src/verify_explanations.py`.

We report for each script a basic template of the command to execute it:
```python
python -m src.explain --dataset <dataset> --model_config <config> --mode <mode> --prefilter <prefilter> --summarization <summarization>
```
```python
python -m src.verify_explanations --dataset <dataset> --model_config <config> --mode <mode> --explanations_path <explanations_path>
```
To inspect the possible values for all the parameters we suggest to run:
```python
python -m src.explain --help
```
and
```python
python -m src.verify_explanations --help
```

We highlight the presence of the value weighted_topology_based in the possible choices for the option --prefilter. It refers to our enhanched Pre-Filter. Moreover, we highlight the option summarization which we introduced with our enhanced Explanation Builder. Specifically, the possible values for summarization are simulation, bisimulation and bisimulation_d1 corresponding to the three alternative formulations of this extension.

Finally we report all the commands executed to obtain the experiments in our paper.

* TransE - DBpedia50 - necessary
    * explanation
        ```python
        python -m src.explain --dataset DBpedia50 --model_config configs/TransE_DBpedia50_explanation.json --mode necessary
        python -m src.explain --dataset DBpedia50 --model_config configs/TransE_DBpedia50_explanation.json --mode necessary --summarization simulation
        python -m src.explain --dataset DBpedia50 --model_config configs/TransE_DBpedia50_explanation.json --mode necessary --summarization bisimulation
        python -m src.explain --dataset DBpedia50 --model_config configs/TransE_DBpedia50_explanation.json --mode necessary --summarization bisimulation_d1
        python -m src.explain --dataset DBpedia50 --model_config configs/TransE_DBpedia50_explanation.json --mode necessary --prefilter weighted_topology_based
        python -m src.explain --dataset DBpedia50 --model_config configs/TransE_DBpedia50_explanation.json --mode necessary --prefilter weighted_topology_based --summarization simulation
        python -m src.explain --dataset DBpedia50 --model_config configs/TransE_DBpedia50_explanation.json --mode necessary --prefilter weighted_topology_based --summarization bisimulation
        python -m src.explain --dataset DBpedia50 --model_config configs/TransE_DBpedia50_explanation.json --mode necessary --prefilter weighted_topology_based --summarization bisimulation_d1
        ```
    * evaluation
        ```python
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/TransE_DBpedia50_evaluation.json --mode necessary --explanations_path results/TransE_DBpedia50_necessary_bfs_th20_no
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/TransE_DBpedia50_evaluation.json --mode necessary --explanations_path results/TransE_DBpedia50_necessary_bfs_th20_simulation
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/TransE_DBpedia50_evaluation.json --mode necessary --explanations_path results/TransE_DBpedia50_necessary_bfs_th20_bisimulation
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/TransE_DBpedia50_evaluation.json --mode necessary --explanations_path results/TransE_DBpedia50_necessary_bfs_th20_bisimulation_d1
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/TransE_DBpedia50_evaluation.json --mode necessary --explanations_path results/TransE_DBpedia50_necessary_wbfs_th20_no
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/TransE_DBpedia50_evaluation.json --mode necessary --explanations_path results/TransE_DBpedia50_necessary_wbfs_th20_simulation
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/TransE_DBpedia50_evaluation.json --mode necessary --explanations_path results/TransE_DBpedia50_necessary_wbfs_th20_bisimulation
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/TransE_DBpedia50_evaluation.json --mode necessary --explanations_path results/TransE_DBpedia50_necessary_wbfs_th20_bisimulation_d1
        ```
* TransE - DBpedia50 - sufficient
    * explanation
        ```python
        python -m src.explain --dataset DBpedia50 --model_config configs/TransE_DBpedia50_explanation.json --mode sufficient
        python -m src.explain --dataset DBpedia50 --model_config configs/TransE_DBpedia50_explanation.json --mode sufficient --summarization simulation
        python -m src.explain --dataset DBpedia50 --model_config configs/TransE_DBpedia50_explanation.json --mode sufficient --summarization bisimulation
        python -m src.explain --dataset DBpedia50 --model_config configs/TransE_DBpedia50_explanation.json --mode sufficient --summarization bisimulation_d1
        python -m src.explain --dataset DBpedia50 --model_config configs/TransE_DBpedia50_explanation.json --mode sufficient --prefilter weighted_topology_based
        python -m src.explain --dataset DBpedia50 --model_config configs/TransE_DBpedia50_explanation.json --mode sufficient --prefilter weighted_topology_based --summarization simulation
        python -m src.explain --dataset DBpedia50 --model_config configs/TransE_DBpedia50_explanation.json --mode sufficient --prefilter weighted_topology_based --summarization bisimulation
        python -m src.explain --dataset DBpedia50 --model_config configs/TransE_DBpedia50_explanation.json --mode sufficient --prefilter weighted_topology_based --summarization bisimulation_d1
        ```
    * evaluation
        ```python
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/TransE_DBpedia50_evaluation.json --mode sufficient --explanations_path results/TransE_DBpedia50_sufficient_bfs_th20_no
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/TransE_DBpedia50_evaluation.json --mode sufficient --explanations_path results/TransE_DBpedia50_sufficient_bfs_th20_simulation
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/TransE_DBpedia50_evaluation.json --mode sufficient --explanations_path results/TransE_DBpedia50_sufficient_bfs_th20_bisimulation
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/TransE_DBpedia50_evaluation.json --mode sufficient --explanations_path results/TransE_DBpedia50_sufficient_bfs_th20_bisimulation_d1
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/TransE_DBpedia50_evaluation.json --mode sufficient --explanations_path results/TransE_DBpedia50_sufficient_wbfs_th20_no
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/TransE_DBpedia50_evaluation.json --mode sufficient --explanations_path results/TransE_DBpedia50_sufficient_wbfs_th20_simulation
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/TransE_DBpedia50_evaluation.json --mode sufficient --explanations_path results/TransE_DBpedia50_sufficient_wbfs_th20_bisimulation
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/TransE_DBpedia50_evaluation.json --mode sufficient --explanations_path results/TransE_DBpedia50_sufficient_wbfs_th20_bisimulation_d1
        ```
* TransE - DB100K - necessary
    * explanation
        ```python
        python -m src.explain --dataset DB100K --model_config configs/TransE_DB100K_explanation.json --mode necessary
        python -m src.explain --dataset DB100K --model_config configs/TransE_DB100K_explanation.json --mode necessary --summarization simulation
        python -m src.explain --dataset DB100K --model_config configs/TransE_DB100K_explanation.json --mode necessary --summarization bisimulation
        python -m src.explain --dataset DB100K --model_config configs/TransE_DB100K_explanation.json --mode necessary --summarization bisimulation_d1
        python -m src.explain --dataset DB100K --model_config configs/TransE_DB100K_explanation.json --mode necessary --prefilter weighted_topology_based
        python -m src.explain --dataset DB100K --model_config configs/TransE_DB100K_explanation.json --mode necessary --prefilter weighted_topology_based --summarization simulation
        python -m src.explain --dataset DB100K --model_config configs/TransE_DB100K_explanation.json --mode necessary --prefilter weighted_topology_based --summarization bisimulation
        python -m src.explain --dataset DB100K --model_config configs/TransE_DB100K_explanation.json --mode necessary --prefilter weighted_topology_based --summarization bisimulation_d1
        ```
    * evaluation
        ```python
        python -m src.verify_explanations --dataset DB100K --model_config configs/TransE_DB100K_evaluation.json --mode necessary --explanations_path results/TransE_DB100K_necessary_bfs_th20_no
        python -m src.verify_explanations --dataset DB100K --model_config configs/TransE_DB100K_evaluation.json --mode necessary --explanations_path results/TransE_DB100K_necessary_bfs_th20_simulation
        python -m src.verify_explanations --dataset DB100K --model_config configs/TransE_DB100K_evaluation.json --mode necessary --explanations_path results/TransE_DB100K_necessary_bfs_th20_bisimulation
        python -m src.verify_explanations --dataset DB100K --model_config configs/TransE_DB100K_evaluation.json --mode necessary --explanations_path results/TransE_DB100K_necessary_bfs_th20_bisimulation_d1
        python -m src.verify_explanations --dataset DB100K --model_config configs/TransE_DB100K_evaluation.json --mode necessary --explanations_path results/TransE_DB100K_necessary_wbfs_th20_no
        python -m src.verify_explanations --dataset DB100K --model_config configs/TransE_DB100K_evaluation.json --mode necessary --explanations_path results/TransE_DB100K_necessary_wbfs_th20_simulation
        python -m src.verify_explanations --dataset DB100K --model_config configs/TransE_DB100K_evaluation.json --mode necessary --explanations_path results/TransE_DB100K_necessary_wbfs_th20_bisimulation
        python -m src.verify_explanations --dataset DB100K --model_config configs/TransE_DB100K_evaluation.json --mode necessary --explanations_path results/TransE_DB100K_necessary_wbfs_th20_bisimulation_d1
        ```
* TransE - DB100K - sufficient
    * explanation
        ```python
        python -m src.explain --dataset DB100K --model_config configs/TransE_DB100K_explanation.json --mode sufficient
        python -m src.explain --dataset DB100K --model_config configs/TransE_DB100K_explanation.json --mode sufficient --summarization simulation
        python -m src.explain --dataset DB100K --model_config configs/TransE_DB100K_explanation.json --mode sufficient --summarization bisimulation
        python -m src.explain --dataset DB100K --model_config configs/TransE_DB100K_explanation.json --mode sufficient --summarization bisimulation_d1
        python -m src.explain --dataset DB100K --model_config configs/TransE_DB100K_explanation.json --mode sufficient --prefilter weighted_topology_based
        python -m src.explain --dataset DB100K --model_config configs/TransE_DB100K_explanation.json --mode sufficient --prefilter weighted_topology_based --summarization simulation
        python -m src.explain --dataset DB100K --model_config configs/TransE_DB100K_explanation.json --mode sufficient --prefilter weighted_topology_based --summarization bisimulation
        python -m src.explain --dataset DB100K --model_config configs/TransE_DB100K_explanation.json --mode sufficient --prefilter weighted_topology_based --summarization bisimulation_d1
        ```
    * evaluation
        ```python
        python -m src.verify_explanations --dataset DB100K --model_config configs/TransE_DB100K_evaluation.json --mode sufficient --explanations_path results/TransE_DB100K_sufficient_bfs_th20_no
        python -m src.verify_explanations --dataset DB100K --model_config configs/TransE_DB100K_evaluation.json --mode sufficient --explanations_path results/TransE_DB100K_sufficient_bfs_th20_simulation
        python -m src.verify_explanations --dataset DB100K --model_config configs/TransE_DB100K_evaluation.json --mode sufficient --explanations_path results/TransE_DB100K_sufficient_bfs_th20_bisimulation
        python -m src.verify_explanations --dataset DB100K --model_config configs/TransE_DB100K_evaluation.json --mode sufficient --explanations_path results/TransE_DB100K_sufficient_bfs_th20_bisimulation_d1
        python -m src.verify_explanations --dataset DB100K --model_config configs/TransE_DB100K_evaluation.json --mode sufficient --explanations_path results/TransE_DB100K_sufficient_wbfs_th20_no
        python -m src.verify_explanations --dataset DB100K --model_config configs/TransE_DB100K_evaluation.json --mode sufficient --explanations_path results/TransE_DB100K_sufficient_wbfs_th20_simulation
        python -m src.verify_explanations --dataset DB100K --model_config configs/TransE_DB100K_evaluation.json --mode sufficient --explanations_path results/TransE_DB100K_sufficient_wbfs_th20_bisimulation
        python -m src.verify_explanations --dataset DB100K --model_config configs/TransE_DB100K_evaluation.json --mode sufficient --explanations_path results/TransE_DB100K_sufficient_wbfs_th20_bisimulation_d1
        ```    
* TransE - YAGO4-20 - necessary
    * explanation
        ```python
        python -m src.explain --dataset YAGO4-20 --model_config configs/TransE_YAGO4-20_explanation.json --mode necessary
        python -m src.explain --dataset YAGO4-20 --model_config configs/TransE_YAGO4-20_explanation.json --mode necessary --summarization simulation
        python -m src.explain --dataset YAGO4-20 --model_config configs/TransE_YAGO4-20_explanation.json --mode necessary --summarization bisimulation
        python -m src.explain --dataset YAGO4-20 --model_config configs/TransE_YAGO4-20_explanation.json --mode necessary --summarization bisimulation_d1
        python -m src.explain --dataset YAGO4-20 --model_config configs/TransE_YAGO4-20_explanation.json --mode necessary --prefilter weighted_topology_based
        python -m src.explain --dataset YAGO4-20 --model_config configs/TransE_YAGO4-20_explanation.json --mode necessary --prefilter weighted_topology_based --summarization simulation
        python -m src.explain --dataset YAGO4-20 --model_config configs/TransE_YAGO4-20_explanation.json --mode necessary --prefilter weighted_topology_based --summarization bisimulation
        python -m src.explain --dataset YAGO4-20 --model_config configs/TransE_YAGO4-20_explanation.json --mode necessary --prefilter weighted_topology_based --summarization bisimulation_d1
        ```
    * evaluation
        ```python
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/TransE_YAGO4-20_evaluation.json --mode necessary --explanations_path results/TransE_DB100K_necessary_bfs_th20_no
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/TransE_YAGO4-20_evaluation.json --mode necessary --explanations_path results/TransE_DB100K_necessary_bfs_th20_simulation
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/TransE_YAGO4-20_evaluation.json --mode necessary --explanations_path results/TransE_DB100K_necessary_bfs_th20_bisimulation
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/TransE_YAGO4-20_evaluation.json --mode necessary --explanations_path results/TransE_DB100K_necessary_bfs_th20_bisimulation_d1
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/TransE_YAGO4-20_evaluation.json --mode necessary --explanations_path results/TransE_DB100K_necessary_wbfs_th20_no
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/TransE_YAGO4-20_evaluation.json --mode necessary --explanations_path results/TransE_DB100K_necessary_wbfs_th20_simulation
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/TransE_YAGO4-20_evaluation.json --mode necessary --explanations_path results/TransE_DB100K_necessary_wbfs_th20_bisimulation
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/TransE_YAGO4-20_evaluation.json --mode necessary --explanations_path results/TransE_DB100K_necessary_wbfs_th20_bisimulation_d1
        ```    
* TransE - YAGO4-20 - sufficient
    * explanation
        ```python
        python -m src.explain --dataset YAGO4-20 --model_config configs/TransE_YAGO4-20_explanation.json --mode sufficient
        python -m src.explain --dataset YAGO4-20 --model_config configs/TransE_YAGO4-20_explanation.json --mode sufficient --summarization simulation
        python -m src.explain --dataset YAGO4-20 --model_config configs/TransE_YAGO4-20_explanation.json --mode sufficient --summarization bisimulation
        python -m src.explain --dataset YAGO4-20 --model_config configs/TransE_YAGO4-20_explanation.json --mode sufficient --summarization bisimulation_d1
        python -m src.explain --dataset YAGO4-20 --model_config configs/TransE_YAGO4-20_explanation.json --mode sufficient --prefilter weighted_topology_based
        python -m src.explain --dataset YAGO4-20 --model_config configs/TransE_YAGO4-20_explanation.json --mode sufficient --prefilter weighted_topology_based --summarization simulation
        python -m src.explain --dataset YAGO4-20 --model_config configs/TransE_YAGO4-20_explanation.json --mode sufficient --prefilter weighted_topology_based --summarization bisimulation
        python -m src.explain --dataset YAGO4-20 --model_config configs/TransE_YAGO4-20_explanation.json --mode sufficient --prefilter weighted_topology_based --summarization bisimulation_d1
        ```
    * evaluation
        ```python
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/TransE_YAGO4-20_evaluation.json --mode sufficient --explanations_path results/TransE_DB100K_sufficient_bfs_th20_no
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/TransE_YAGO4-20_evaluation.json --mode sufficient --explanations_path results/TransE_DB100K_sufficient_bfs_th20_simulation
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/TransE_YAGO4-20_evaluation.json --mode sufficient --explanations_path results/TransE_DB100K_sufficient_bfs_th20_bisimulation
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/TransE_YAGO4-20_evaluation.json --mode sufficient --explanations_path results/TransE_DB100K_sufficient_bfs_th20_bisimulation_d1
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/TransE_YAGO4-20_evaluation.json --mode sufficient --explanations_path results/TransE_DB100K_sufficient_wbfs_th20_no
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/TransE_YAGO4-20_evaluation.json --mode sufficient --explanations_path results/TransE_DB100K_sufficient_wbfs_th20_simulation
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/TransE_YAGO4-20_evaluation.json --mode sufficient --explanations_path results/TransE_DB100K_sufficient_wbfs_th20_bisimulation
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/TransE_YAGO4-20_evaluation.json --mode sufficient --explanations_path results/TransE_DB100K_sufficient_wbfs_th20_bisimulation_d1
        ```
* ConvE - DBpedia50 - necessary
    * explanation
        ```python
        python -m src.explain --dataset DBpedia50 --model_config configs/ConvE_DBpedia50_explanation.json --mode necessary
        python -m src.explain --dataset DBpedia50 --model_config configs/ConvE_DBpedia50_explanation.json --mode necessary --summarization simulation
        python -m src.explain --dataset DBpedia50 --model_config configs/ConvE_DBpedia50_explanation.json --mode necessary --summarization bisimulation
        python -m src.explain --dataset DBpedia50 --model_config configs/ConvE_DBpedia50_explanation.json --mode necessary --summarization bisimulation_d1
        python -m src.explain --dataset DBpedia50 --model_config configs/ConvE_DBpedia50_explanation.json --mode necessary --prefilter weighted_topology_based
        python -m src.explain --dataset DBpedia50 --model_config configs/ConvE_DBpedia50_explanation.json --mode necessary --prefilter weighted_topology_based --summarization simulation
        python -m src.explain --dataset DBpedia50 --model_config configs/ConvE_DBpedia50_explanation.json --mode necessary --prefilter weighted_topology_based --summarization bisimulation
        python -m src.explain --dataset DBpedia50 --model_config configs/ConvE_DBpedia50_explanation.json --mode necessary --prefilter weighted_topology_based --summarization bisimulation_d1
        ```
    * evaluation
        ```python
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/ConvE_DBpedia50_explanation.json --mode necessary --explanations_path results/ConvE_DBpedia50_necessary_bfs_th20_no
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/ConvE_DBpedia50_explanation.json --mode necessary --explanations_path results/ConvE_DBpedia50_necessary_bfs_th20_simulation
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/ConvE_DBpedia50_explanation.json --mode necessary --explanations_path results/ConvE_DBpedia50_necessary_bfs_th20_bisimulation
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/ConvE_DBpedia50_explanation.json --mode necessary --explanations_path results/ConvE_DBpedia50_necessary_bfs_th20_bisimulation_d1
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/ConvE_DBpedia50_explanation.json --mode necessary --explanations_path results/ConvE_DBpedia50_necessary_wbfs_th20_no
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/ConvE_DBpedia50_explanation.json --mode necessary --explanations_path results/ConvE_DBpedia50_necessary_wbfs_th20_simulation
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/ConvE_DBpedia50_explanation.json --mode necessary --explanations_path results/ConvE_DBpedia50_necessary_wbfs_th20_bisimulation
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/ConvE_DBpedia50_explanation.json --mode necessary --explanations_path results/ConvE_DBpedia50_necessary_wbfs_th20_bisimulation_d1
        ```
* ConvE - DBpedia50 - sufficient
    * explanation
        ```python
        python -m src.explain --dataset DBpedia50 --model_config configs/ConvE_DBpedia50_explanation.json --mode sufficient
        python -m src.explain --dataset DBpedia50 --model_config configs/ConvE_DBpedia50_explanation.json --mode sufficient --summarization simulation
        python -m src.explain --dataset DBpedia50 --model_config configs/ConvE_DBpedia50_explanation.json --mode sufficient --summarization bisimulation
        python -m src.explain --dataset DBpedia50 --model_config configs/ConvE_DBpedia50_explanation.json --mode sufficient --summarization bisimulation_d1
        python -m src.explain --dataset DBpedia50 --model_config configs/ConvE_DBpedia50_explanation.json --mode sufficient --prefilter weighted_topology_based
        python -m src.explain --dataset DBpedia50 --model_config configs/ConvE_DBpedia50_explanation.json --mode sufficient --prefilter weighted_topology_based --summarization simulation
        python -m src.explain --dataset DBpedia50 --model_config configs/ConvE_DBpedia50_explanation.json --mode sufficient --prefilter weighted_topology_based --summarization bisimulation
        python -m src.explain --dataset DBpedia50 --model_config configs/ConvE_DBpedia50_explanation.json --mode sufficient --prefilter weighted_topology_based --summarization bisimulation_d1
        ```
    * evaluation
        ```python
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/ConvE_DBpedia50_explanation.json --mode sufficient --explanations_path results/ConvE_DBpedia50_sufficient_bfs_th20_no
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/ConvE_DBpedia50_explanation.json --mode sufficient --explanations_path results/ConvE_DBpedia50_sufficient_bfs_th20_simulation
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/ConvE_DBpedia50_explanation.json --mode sufficient --explanations_path results/ConvE_DBpedia50_sufficient_bfs_th20_bisimulation
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/ConvE_DBpedia50_explanation.json --mode sufficient --explanations_path results/ConvE_DBpedia50_sufficient_bfs_th20_bisimulation_d1
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/ConvE_DBpedia50_explanation.json --mode sufficient --explanations_path results/ConvE_DBpedia50_sufficient_wbfs_th20_no
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/ConvE_DBpedia50_explanation.json --mode sufficient --explanations_path results/ConvE_DBpedia50_sufficient_wbfs_th20_simulation
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/ConvE_DBpedia50_explanation.json --mode sufficient --explanations_path results/ConvE_DBpedia50_sufficient_wbfs_th20_bisimulation
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/ConvE_DBpedia50_explanation.json --mode sufficient --explanations_path results/ConvE_DBpedia50_sufficient_wbfs_th20_bisimulation_d1
        ```
* ConvE - DB100K - necessary
    * explanation
        ```python
        python -m src.explain --dataset DB100K --model_config configs/ConvE_DB100K_explanation.json --mode necessary
        python -m src.explain --dataset DB100K --model_config configs/ConvE_DB100K_explanation.json --mode necessary --summarization simulation
        python -m src.explain --dataset DB100K --model_config configs/ConvE_DB100K_explanation.json --mode necessary --summarization bisimulation
        python -m src.explain --dataset DB100K --model_config configs/ConvE_DB100K_explanation.json --mode necessary --summarization bisimulation_d1
        python -m src.explain --dataset DB100K --model_config configs/ConvE_DB100K_explanation.json --mode necessary --prefilter weighted_topology_based
        python -m src.explain --dataset DB100K --model_config configs/ConvE_DB100K_explanation.json --mode necessary --prefilter weighted_topology_based --summarization simulation
        python -m src.explain --dataset DB100K --model_config configs/ConvE_DB100K_explanation.json --mode necessary --prefilter weighted_topology_based --summarization bisimulation
        python -m src.explain --dataset DB100K --model_config configs/TransE_DB100K_explanation.json --mode necessary --prefilter weighted_topology_based --summarization bisimulation_d1
        ```
    * evaluation
        ```python
        python -m src.verify_explanations --dataset DB100K --model_config configs/ConvE_DB100K_explanation.json --mode necessary --explanations_path results/ConvE_DB100K_necessary_bfs_th20_no
        python -m src.verify_explanations --dataset DB100K --model_config configs/ConvE_DB100K_explanation.json --mode necessary --explanations_path results/ConvE_DB100K_necessary_bfs_th20_simulation
        python -m src.verify_explanations --dataset DB100K --model_config configs/ConvE_DB100K_explanation.json --mode necessary --explanations_path results/ConvE_DB100K_necessary_bfs_th20_bisimulation
        python -m src.verify_explanations --dataset DB100K --model_config configs/ConvE_DB100K_explanation.json --mode necessary --explanations_path results/ConvE_DB100K_necessary_bfs_th20_bisimulation_d1
        python -m src.verify_explanations --dataset DB100K --model_config configs/ConvE_DB100K_explanation.json --mode necessary --explanations_path results/ConvE_DB100K_necessary_wbfs_th20_no
        python -m src.verify_explanations --dataset DB100K --model_config configs/ConvE_DB100K_explanation.json --mode necessary --explanations_path results/ConvE_DB100K_necessary_wbfs_th20_simulation
        python -m src.verify_explanations --dataset DB100K --model_config configs/ConvE_DB100K_explanation.json --mode necessary --explanations_path results/ConvE_DB100K_necessary_wbfs_th20_bisimulation
        python -m src.verify_explanations --dataset DB100K --model_config configs/ConvE_DB100K_explanation.json --mode necessary --explanations_path results/ConvE_DB100K_necessary_wbfs_th20_bisimulation_d1
        ```
* ConvE - DB100K - sufficient
    * explanation
        ```python
        python -m src.explain --dataset DB100K --model_config configs/ConvE_DB100K_explanation.json --mode sufficient
        python -m src.explain --dataset DB100K --model_config configs/ConvE_DB100K_explanation.json --mode sufficient --summarization simulation
        python -m src.explain --dataset DB100K --model_config configs/ConvE_DB100K_explanation.json --mode sufficient --summarization bisimulation
        python -m src.explain --dataset DB100K --model_config configs/ConvE_DB100K_explanation.json --mode sufficient --summarization bisimulation_d1
        python -m src.explain --dataset DB100K --model_config configs/ConvE_DB100K_explanation.json --mode sufficient --prefilter weighted_topology_based
        python -m src.explain --dataset DB100K --model_config configs/ConvE_DB100K_explanation.json --mode sufficient --prefilter weighted_topology_based --summarization simulation
        python -m src.explain --dataset DB100K --model_config configs/ConvE_DB100K_explanation.json --mode sufficient --prefilter weighted_topology_based --summarization bisimulation
        python -m src.explain --dataset DB100K --model_config configs/ConvE_DB100K_explanation.json --mode sufficient --prefilter weighted_topology_based --summarization bisimulation_d1
        ```
    * evaluation
        ```python
        python -m src.verify_explanations --dataset DB100K --model_config configs/ConvE_DB100K_explanation.json --mode sufficient --explanations_path results/ConvE_DB100K_sufficient_bfs_th20_no
        python -m src.verify_explanations --dataset DB100K --model_config configs/ConvE_DB100K_explanation.json --mode sufficient --explanations_path results/ConvE_DB100K_sufficient_bfs_th20_simulation
        python -m src.verify_explanations --dataset DB100K --model_config configs/ConvE_DB100K_explanation.json --mode sufficient --explanations_path results/ConvE_DB100K_sufficient_bfs_th20_bisimulation
        python -m src.verify_explanations --dataset DB100K --model_config configs/ConvE_DB100K_explanation.json --mode sufficient --explanations_path results/ConvE_DB100K_sufficient_bfs_th20_bisimulation_d1
        python -m src.verify_explanations --dataset DB100K --model_config configs/ConvE_DB100K_explanation.json --mode sufficient --explanations_path results/ConvE_DB100K_sufficient_wbfs_th20_no
        python -m src.verify_explanations --dataset DB100K --model_config configs/ConvE_DB100K_explanation.json --mode sufficient --explanations_path results/ConvE_DB100K_sufficient_wbfs_th20_simulation
        python -m src.verify_explanations --dataset DB100K --model_config configs/ConvE_DB100K_explanation.json --mode sufficient --explanations_path results/ConvE_DB100K_sufficient_wbfs_th20_bisimulation
        python -m src.verify_explanations --dataset DB100K --model_config configs/ConvE_DB100K_explanation.json --mode sufficient --explanations_path results/ConvE_DB100K_sufficient_wbfs_th20_bisimulation_d1
        ```    
* ConvE - YAGO4-20 - necessary
    * explanation
        ```python
        python -m src.explain --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_training.json --mode necessary
        python -m src.explain --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_training.json --mode necessary --summarization simulation
        python -m src.explain --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_training.json --mode necessary --summarization bisimulation
        python -m src.explain --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_training.json --mode necessary --summarization bisimulation_d1
        python -m src.explain --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_training.json --mode necessary --prefilter weighted_topology_based
        python -m src.explain --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_training.json --mode necessary --prefilter weighted_topology_based --summarization simulation
        python -m src.explain --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_training.json --mode necessary --prefilter weighted_topology_based --summarization bisimulation
        python -m src.explain --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_training.json --mode necessary --prefilter weighted_topology_based --summarization bisimulation_d1
        ```
    * evaluation
        ```python
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_training.json --mode necessary --explanations_path results/ConvE_YAGO4-20_necessary_bfs_th20_no
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_training.json --mode necessary --explanations_path results/ConvE_YAGO4-20_necessary_bfs_th20_simulation
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_training.json --mode necessary --explanations_path results/ConvE_YAGO4-20_necessary_bfs_th20_bisimulation
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_training.json --mode necessary --explanations_path results/ConvE_YAGO4-20_necessary_bfs_th20_bisimulation_d1
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_training.json --mode necessary --explanations_path results/ConvE_YAGO4-20_necessary_wbfs_th20_no
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_training.json --mode necessary --explanations_path results/ConvE_YAGO4-20_necessary_wbfs_th20_simulation
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_training.json --mode necessary --explanations_path results/ConvE_YAGO4-20_necessary_wbfs_th20_bisimulation
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_training.json --mode necessary --explanations_path results/ConvE_YAGO4-20_necessary_wbfs_th20_bisimulation_d1
        ```    
* ConvE - YAGO4-20 - sufficient
    * explanation
        ```python
        python -m src.explain --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_training.json --mode sufficient
        python -m src.explain --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_training.json --mode sufficient --summarization simulation
        python -m src.explain --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_training.json --mode sufficient --summarization bisimulation
        python -m src.explain --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_training.json --mode sufficient --summarization bisimulation_d1
        python -m src.explain --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_training.json --mode sufficient --prefilter weighted_topology_based
        python -m src.explain --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_training.json --mode sufficient --prefilter weighted_topology_based --summarization simulation
        python -m src.explain --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_training.json --mode sufficient --prefilter weighted_topology_based --summarization bisimulation
        python -m src.explain --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_training.json --mode sufficient --prefilter weighted_topology_based --summarization bisimulation_d1
        ```
    * evaluation
        ```python
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_explanation.json --mode sufficient --explanations_path results/ConvE_YAGO4-20_sufficient_bfs_th20_no
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_explanation.json --mode sufficient --explanations_path results/ConvE_YAGO4-20_sufficient_bfs_th20_simulation
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_explanation.json --mode sufficient --explanations_path results/ConvE_YAGO4-20_sufficient_bfs_th20_bisimulation
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_explanation.json --mode sufficient --explanations_path results/ConvE_YAGO4-20_sufficient_bfs_th20_bisimulation_d1
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_explanation.json --mode sufficient --explanations_path results/ConvE_YAGO4-20_sufficient_wbfs_th20_no
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_explanation.json --mode sufficient --explanations_path results/ConvE_YAGO4-20_sufficient_wbfs_th20_simulation
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_explanation.json --mode sufficient --explanations_path results/ConvE_YAGO4-20_sufficient_wbfs_th20_bisimulation
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_explanation.json --mode sufficient --explanations_path results/ConvE_YAGO4-20_sufficient_wbfs_th20_bisimulation_d1
        ```
* ComplEx - DBpedia50 - necessary
    * explanation
        ```python
        python -m src.explain --dataset DBpedia50 --model_config configs/ComplEx_DBpedia50_explanation.json --mode necessary
        python -m src.explain --dataset DBpedia50 --model_config configs/ComplEx_DBpedia50_explanation.json --mode necessary --summarization simulation
        python -m src.explain --dataset DBpedia50 --model_config configs/ComplEx_DBpedia50_explanation.json --mode necessary --summarization bisimulation
        python -m src.explain --dataset DBpedia50 --model_config configs/ComplEx_DBpedia50_explanation.json --mode necessary --summarization bisimulation_d1
        python -m src.explain --dataset DBpedia50 --model_config configs/ComplEx_DBpedia50_explanation.json --mode necessary --prefilter weighted_topology_based
        python -m src.explain --dataset DBpedia50 --model_config configs/ComplEx_DBpedia50_explanation.json --mode necessary --prefilter weighted_topology_based --summarization simulation
        python -m src.explain --dataset DBpedia50 --model_config configs/ComplEx_DBpedia50_explanation.json --mode necessary --prefilter weighted_topology_based --summarization bisimulation
        python -m src.explain --dataset DBpedia50 --model_config configs/ComplEx_DBpedia50_explanation.json --mode necessary --prefilter weighted_topology_based --summarization bisimulation_d1
        ```
    * evaluation
        ```python
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/ComplEx_DBpedia50_explanation.json --mode necessary --explanations_path results/ComplEx_DBpedia50_necessary_bfs_th20_no
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/ComplEx_DBpedia50_explanation.json --mode necessary --explanations_path results/ComplEx_DBpedia50_necessary_bfs_th20_simulation
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/ComplEx_DBpedia50_explanation.json --mode necessary --explanations_path results/ComplEx_DBpedia50_necessary_bfs_th20_bisimulation
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/ComplEx_DBpedia50_explanation.json --mode necessary --explanations_path results/ComplEx_DBpedia50_necessary_bfs_th20_bisimulation_d1
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/ComplEx_DBpedia50_explanation.json --mode necessary --explanations_path results/ComplEx_DBpedia50_necessary_wbfs_th20_no
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/ComplEx_DBpedia50_explanation.json --mode necessary --explanations_path results/ComplEx_DBpedia50_necessary_wbfs_th20_simulation
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/ComplEx_DBpedia50_explanation.json --mode necessary --explanations_path results/ComplEx_DBpedia50_necessary_wbfs_th20_bisimulation
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/ComplEx_DBpedia50_explanation.json --mode necessary --explanations_path results/ComplEx_DBpedia50_necessary_wbfs_th20_bisimulation_d1
        ```
* ComplEx - DBpedia50 - sufficient
    * explanation
        ```python
        python -m src.explain --dataset DBpedia50 --model_config configs/ComplEx_DBpedia50_explanation.json --mode sufficient
        python -m src.explain --dataset DBpedia50 --model_config configs/ComplEx_DBpedia50_explanation.json --mode sufficient --summarization simulation
        python -m src.explain --dataset DBpedia50 --model_config configs/ComplEx_DBpedia50_explanation.json --mode sufficient --summarization bisimulation
        python -m src.explain --dataset DBpedia50 --model_config configs/ComplEx_DBpedia50_explanation.json --mode sufficient --summarization bisimulation_d1
        python -m src.explain --dataset DBpedia50 --model_config configs/ComplEx_DBpedia50_explanation.json --mode sufficient --prefilter weighted_topology_based
        python -m src.explain --dataset DBpedia50 --model_config configs/ComplEx_DBpedia50_explanation.json --mode sufficient --prefilter weighted_topology_based --summarization simulation
        python -m src.explain --dataset DBpedia50 --model_config configs/ComplEx_DBpedia50_explanation.json --mode sufficient --prefilter weighted_topology_based --summarization bisimulation
        python -m src.explain --dataset DBpedia50 --model_config configs/ComplEx_DBpedia50_explanation.json --mode sufficient --prefilter weighted_topology_based --summarization bisimulation_d1
        ```
    * evaluation
        ```python
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/ComplEx_DBpedia50_explanation.json --mode sufficient --explanations_path results/ComplEx_DBpedia50_sufficient_bfs_th20_no
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/ComplEx_DBpedia50_explanation.json --mode sufficient --explanations_path results/ComplEx_DBpedia50_sufficient_bfs_th20_simulation
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/ComplEx_DBpedia50_explanation.json --mode sufficient --explanations_path results/ComplEx_DBpedia50_sufficient_bfs_th20_bisimulation
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/ComplEx_DBpedia50_explanation.json --mode sufficient --explanations_path results/ComplEx_DBpedia50_sufficient_bfs_th20_bisimulation_d1
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/ComplEx_DBpedia50_explanation.json --mode sufficient --explanations_path results/ComplEx_DBpedia50_sufficient_wbfs_th20_no
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/ComplEx_DBpedia50_explanation.json --mode sufficient --explanations_path results/ComplEx_DBpedia50_sufficient_wbfs_th20_simulation
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/ComplEx_DBpedia50_explanation.json --mode sufficient --explanations_path results/ComplEx_DBpedia50_sufficient_wbfs_th20_bisimulation
        python -m src.verify_explanations --dataset DBpedia50 --model_config configs/ComplEx_DBpedia50_explanation.json --mode sufficient --explanations_path results/ComplEx_DBpedia50_sufficient_wbfs_th20_bisimulation_d1
        ```
* ComplEx - DB100K - necessary
    * explanation
        ```python
        python -m src.explain --dataset DB100K --model_config configs/ComplEx_DB100K_explanation.json --mode necessary
        python -m src.explain --dataset DB100K --model_config configs/ComplEx_DB100K_explanation.json --mode necessary --summarization simulation
        python -m src.explain --dataset DB100K --model_config configs/ComplEx_DB100K_explanation.json --mode necessary --summarization bisimulation
        python -m src.explain --dataset DB100K --model_config configs/ComplEx_DB100K_explanation.json --mode necessary --summarization bisimulation_d1
        python -m src.explain --dataset DB100K --model_config configs/ComplEx_DB100K_explanation.json --mode necessary --prefilter weighted_topology_based
        python -m src.explain --dataset DB100K --model_config configs/ComplEx_DB100K_explanation.json --mode necessary --prefilter weighted_topology_based --summarization simulation
        python -m src.explain --dataset DB100K --model_config configs/ComplEx_DB100K_explanation.json --mode necessary --prefilter weighted_topology_based --summarization bisimulation
        python -m src.explain --dataset DB100K --model_config configs/ComplEx_DB100K_explanation.json --mode necessary --prefilter weighted_topology_based --summarization bisimulation_d1
        ```
    * evaluation
        ```python
        python -m src.verify_explanations --dataset DB100K --model_config configs/ComplEx_DB100K_explanation.json --mode necessary --explanations_path results/ComplEx_DB100K_necessary_bfs_th20_no
        python -m src.verify_explanations --dataset DB100K --model_config configs/ComplEx_DB100K_explanation.json --mode necessary --explanations_path results/ComplEx_DB100K_necessary_bfs_th20_simulation
        python -m src.verify_explanations --dataset DB100K --model_config configs/ComplEx_DB100K_explanation.json --mode necessary --explanations_path results/ComplEx_DB100K_necessary_bfs_th20_bisimulation
        python -m src.verify_explanations --dataset DB100K --model_config configs/ComplEx_DB100K_explanation.json --mode necessary --explanations_path results/ComplEx_DB100K_necessary_bfs_th20_bisimulation_d1
        python -m src.verify_explanations --dataset DB100K --model_config configs/ComplEx_DB100K_explanation.json --mode necessary --explanations_path results/ComplEx_DB100K_necessary_wbfs_th20_no
        python -m src.verify_explanations --dataset DB100K --model_config configs/ComplEx_DB100K_explanation.json --mode necessary --explanations_path results/ComplEx_DB100K_necessary_wbfs_th20_simulation
        python -m src.verify_explanations --dataset DB100K --model_config configs/ComplEx_DB100K_explanation.json --mode necessary --explanations_path results/ComplEx_DB100K_necessary_wbfs_th20_bisimulation
        python -m src.verify_explanations --dataset DB100K --model_config configs/ComplEx_DB100K_explanation.json --mode necessary --explanations_path results/ComplEx_DB100K_necessary_wbfs_th20_bisimulation_d1
        ```
* ComplEx - DB100K - sufficient
    * explanation
        ```python
        python -m src.explain --dataset DB100K --model_config configs/ComplEx_DB100K_explanation.json --mode sufficient
        python -m src.explain --dataset DB100K --model_config configs/ComplEx_DB100K_explanation.json --mode sufficient --summarization simulation
        python -m src.explain --dataset DB100K --model_config configs/ComplEx_DB100K_explanation.json --mode sufficient --summarization bisimulation
        python -m src.explain --dataset DB100K --model_config configs/ComplEx_DB100K_explanation.json --mode sufficient --summarization bisimulation_d1
        python -m src.explain --dataset DB100K --model_config configs/ComplEx_DB100K_explanation.json --mode sufficient --prefilter weighted_topology_based
        python -m src.explain --dataset DB100K --model_config configs/ComplEx_DB100K_explanation.json --mode sufficient --prefilter weighted_topology_based --summarization simulation
        python -m src.explain --dataset DB100K --model_config configs/ComplEx_DB100K_explanation.json --mode sufficient --prefilter weighted_topology_based --summarization bisimulation
        python -m src.explain --dataset DB100K --model_config configs/ComplEx_DB100K_explanation.json --mode sufficient --prefilter weighted_topology_based --summarization bisimulation_d1
        ```
    * evaluation
        ```python
        python -m src.verify_explanations --dataset DB100K --model_config configs/ComplEx_DB100K_explanation.json --mode sufficient --explanations_path results/ComplEx_DB100K_sufficient_bfs_th20_no
        python -m src.verify_explanations --dataset DB100K --model_config configs/ComplEx_DB100K_explanation.json --mode sufficient --explanations_path results/ComplEx_DB100K_sufficient_bfs_th20_simulation
        python -m src.verify_explanations --dataset DB100K --model_config configs/ComplEx_DB100K_explanation.json --mode sufficient --explanations_path results/ComplEx_DB100K_sufficient_bfs_th20_bisimulation
        python -m src.verify_explanations --dataset DB100K --model_config configs/ComplEx_DB100K_explanation.json --mode sufficient --explanations_path results/ComplEx_DB100K_sufficient_bfs_th20_bisimulation_d1
        python -m src.verify_explanations --dataset DB100K --model_config configs/ComplEx_DB100K_explanation.json --mode sufficient --explanations_path results/ComplEx_DB100K_sufficient_wbfs_th20_no
        python -m src.verify_explanations --dataset DB100K --model_config configs/ComplEx_DB100K_explanation.json --mode sufficient --explanations_path results/ComplEx_DB100K_sufficient_wbfs_th20_simulation
        python -m src.verify_explanations --dataset DB100K --model_config configs/ComplEx_DB100K_explanation.json --mode sufficient --explanations_path results/ComplEx_DB100K_sufficient_wbfs_th20_bisimulation
        python -m src.verify_explanations --dataset DB100K --model_config configs/ComplEx_DB100K_explanation.json --mode sufficient --explanations_path results/ComplEx_DB100K_sufficient_wbfs_th20_bisimulation_d1
        ```    
* ComplEx - YAGO4-20 - necessary
    * explanation
        ```python
        python -m src.explain --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_explanation.json --mode necessary
        python -m src.explain --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_explanation.json --mode necessary --summarization simulation
        python -m src.explain --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_explanation.json --mode necessary --summarization bisimulation
        python -m src.explain --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_explanation.json --mode necessary --summarization bisimulation_d1
        python -m src.explain --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_explanation.json --mode necessary --prefilter weighted_topology_based
        python -m src.explain --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_explanation.json --mode necessary --prefilter weighted_topology_based --summarization simulation
        python -m src.explain --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_explanation.json --mode necessary --prefilter weighted_topology_based --summarization bisimulation
        python -m src.explain --dataset YAGO4-20 --model_config configs/ConvE_YAGO4-20_explanation.json --mode necessary --prefilter weighted_topology_based --summarization bisimulation_d1
        ```
    * evaluation
        ```python
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/ComplEx_YAGO4-20_explanation.json --mode necessary --explanations_path results/ComplEx_YAGO4-20_necessary_bfs_th20_no
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/ComplEx_YAGO4-20_explanation.json --mode necessary --explanations_path results/ComplEx_YAGO4-20_necessary_bfs_th20_simulation
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/ComplEx_YAGO4-20_explanation.json --mode necessary --explanations_path results/ComplEx_YAGO4-20_necessary_bfs_th20_bisimulation
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/ComplEx_YAGO4-20_explanation.json --mode necessary --explanations_path results/ComplEx_YAGO4-20_necessary_bfs_th20_bisimulation_d1
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/ComplEx_YAGO4-20_explanation.json --mode necessary --explanations_path results/ComplEx_YAGO4-20_necessary_wbfs_th20_no
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/ComplEx_YAGO4-20_explanation.json --mode necessary --explanations_path results/ComplEx_YAGO4-20_necessary_wbfs_th20_simulation
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/ComplEx_YAGO4-20_explanation.json --mode necessary --explanations_path results/ComplEx_YAGO4-20_necessary_wbfs_th20_bisimulation
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/ComplEx_YAGO4-20_explanation.json --mode necessary --explanations_path results/ComplEx_YAGO4-20_necessary_wbfs_th20_bisimulation_d1
        ```    
* ComplEx - YAGO4-20 - sufficient
    * explanation
        ```python
        python -m src.explain --dataset YAGO4-20 --model_config configs/ComplEx_YAGO4-20_explanation.json --mode sufficient
        python -m src.explain --dataset YAGO4-20 --model_config configs/ComplEx_YAGO4-20_explanation.json --mode sufficient --summarization simulation
        python -m src.explain --dataset YAGO4-20 --model_config configs/ComplEx_YAGO4-20_explanation.json --mode sufficient --summarization bisimulation
        python -m src.explain --dataset YAGO4-20 --model_config configs/ComplEx_YAGO4-20_explanation.json --mode sufficient --summarization bisimulation_d1
        python -m src.explain --dataset YAGO4-20 --model_config configs/ComplEx_YAGO4-20_explanation.json --mode sufficient --prefilter weighted_topology_based
        python -m src.explain --dataset YAGO4-20 --model_config configs/ComplEx_YAGO4-20_explanation.json --mode sufficient --prefilter weighted_topology_based --summarization simulation
        python -m src.explain --dataset YAGO4-20 --model_config configs/ComplEx_YAGO4-20_explanation.json --mode sufficient --prefilter weighted_topology_based --summarization bisimulation
        python -m src.explain --dataset YAGO4-20 --model_config configs/ComplEx_YAGO4-20_explanation.json --mode sufficient --prefilter weighted_topology_based --summarization bisimulation_d1
        ```
    * evaluation
        ```python
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/ComplEx_YAGO4-20_explanation.json --mode sufficient --explanations_path results/ComplEx_YAGO4-20_sufficient_bfs_th20_no
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/ComplEx_YAGO4-20_explanation.json --mode sufficient --explanations_path results/ComplEx_YAGO4-20_sufficient_bfs_th20_simulation
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/ComplEx_YAGO4-20_explanation.json --mode sufficient --explanations_path results/ComplEx_YAGO4-20_sufficient_bfs_th20_bisimulation
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/ComplEx_YAGO4-20_explanation.json --mode sufficient --explanations_path results/ComplEx_YAGO4-20_sufficient_bfs_th20_bisimulation_d1
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/ComplEx_YAGO4-20_explanation.json --mode sufficient --explanations_path results/ComplEx_YAGO4-20_sufficient_wbfs_th20_no
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/ComplEx_YAGO4-20_explanation.json --mode sufficient --explanations_path results/ComplEx_YAGO4-20_sufficient_wbfs_th20_simulation
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/ComplEx_YAGO4-20_explanation.json --mode sufficient --explanations_path results/ComplEx_YAGO4-20_sufficient_wbfs_th20_bisimulation
        python -m src.verify_explanations --dataset YAGO4-20 --model_config configs/ComplEx_YAGO4-20_explanation.json --mode sufficient --explanations_path results/ComplEx_YAGO4-20_sufficient_wbfs_th20_bisimulation_d1
        ```   
