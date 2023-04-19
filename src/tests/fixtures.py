import pytest
import torch

from src.dataset import Dataset

from src.link_prediction.models import ComplEx
from src.link_prediction.models import (
    LEARNING_RATE,
    DIMENSION,
    INIT_SCALE,
    OPTIMIZER_NAME,
    DECAY_1,
    DECAY_2,
    REGULARIZER_WEIGHT,
    EPOCHS,
    BATCH_SIZE,
    REGULARIZER_NAME,
)


@pytest.fixture
def sample_to_explain(dataset):
    return dataset.fact_to_sample(("/m/03hkch7", "/film/film/genre", "/m/0219x_"))


@pytest.fixture
def dataset():
    return Dataset(name="FB15k-237", separator="\t", load=True)


@pytest.fixture
def hyperparameters():
    return {
        DIMENSION: 1000,
        INIT_SCALE: 1e-3,
        LEARNING_RATE: 0.01,
        OPTIMIZER_NAME: "Adagrad",
        DECAY_1: 0.9,
        DECAY_2: 0.999,
        REGULARIZER_WEIGHT: 2.5e-3,
        EPOCHS: 200,
        BATCH_SIZE: 100,
        REGULARIZER_NAME: "N3",
    }


@pytest.fixture
def model(dataset, hyperparameters):
    model = ComplEx(
        dataset=dataset,
        hyperparameters=hyperparameters,
        init_random=True,
    )

    model.to("cuda")
    model.load_state_dict(torch.load("./stored_models/ComplEx_FB15k-237.pt"))
    model.eval()

    return model
