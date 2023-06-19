import json
import pytest

import torch

from src.data import Dataset

from src.link_prediction.models import ComplEx, ComplExHyperParams


@pytest.fixture
def dataset():
    return Dataset(dataset="FB15k-237")


@pytest.fixture
def model_config():
    return json.load(open("configs/complex.json", "r"))


@pytest.fixture
def model_params(model_config):
    return model_config["model_params"]


@pytest.fixture
def training_params(model_config):
    return model_config["training"]


@pytest.fixture
def model(dataset, model_params):
    hp = ComplExHyperParams(**model_params)
    model = ComplEx(dataset=dataset, hp=hp, init_random=True)

    model.to("cuda")
    model.load_state_dict(torch.load("./models/ComplEx_FB15k-237.pt"))
    model.eval()

    return model


@pytest.fixture
def pred(dataset):
    return dataset.ids_triple(
        ("/m/07z2lx", "/award/award_category/category_of", "/m/0gcf2r")
    )
