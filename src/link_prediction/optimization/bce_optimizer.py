import torch
import numpy as np

from collections import defaultdict
from pydantic import BaseModel
from tqdm import tqdm

from torch import optim

from .optimizer import Optimizer

from ..models import Model
from ..models import ConvE


class BCEOptimizerHyperParams(BaseModel):
    batch_size: int
    label_smoothing: float
    lr: float
    decay: float
    epochs: int


class BCEOptimizer(Optimizer):
    def __init__(self, model: Model, hp: BCEOptimizerHyperParams, verbose: bool = True):
        super().__init__(model=model, hp=hp, verbose=verbose)

        self.batch_size = hp.batch_size
        self.label_smoothing = hp.label_smoothing
        self.lr = hp.lr
        self.decay = hp.decay
        self.epochs = hp.epochs

        self.loss = torch.nn.BCELoss()
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.decay)

    def get_hyperparams_class():
        return BCEOptimizerHyperParams

    def get_kelpie_class():
        return KelpieBCEOptimizer

    def train(
        self,
        training_triples: np.array,
        save_path: str = None,
        eval_every: int = -1,
        valid_triples: np.array = None,
    ):
        inverse_triples = self.model.dataset.invert_triples(training_triples)

        training_triples = np.vstack((training_triples, inverse_triples))
        er_vocab = self.extract_er_vocab(training_triples)
        er_vocab_pairs = list(er_vocab.keys())

        self.model.cuda()

        for e in tqdm(range(1, self.epochs + 1)):
            self.epoch(er_vocab, er_vocab_pairs, self.batch_size)

            is_eval_epoch = eval_every > 0 and e % eval_every == 0
            if valid_triples is not None and is_eval_epoch:
                self.model.eval()
                metrics = self.evaluator.evaluate(valid_triples, write_output=False)

                print(f"\tValidation Hits@1: {metrics['h1']}")
                print(f"\tValidation Hits@10: {metrics['h10']}")
                print(f"\tValidation Mean Reciprocal Rank: {metrics['mrr']}")
                print(f"\tValidation Mean Rank: {metrics['mr']}")

                if save_path is not None:
                    print("\t Saving model...")
                    torch.save(self.model.state_dict(), save_path)

        if save_path is not None:
            print("\t Saving model...")
            torch.save(self.model.state_dict(), save_path)

    def extract_er_vocab(self, triples):
        er_vocab = defaultdict(list)
        for triple in triples:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def extract_batch(self, er_vocab, er_vocab_pairs, batch_start, batch_size):
        batch = er_vocab_pairs[
            batch_start : min(batch_start + batch_size, len(er_vocab_pairs))
        ]

        targets = np.zeros((len(batch), self.dataset.num_entities))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.0
        if self.label_smoothing:
            targets = ((1.0 - self.label_smoothing) * targets) + (
                1.0 / targets.shape[1]
            )

        return torch.tensor(np.array(batch)).cuda(), torch.FloatTensor(targets).cuda()

    def epoch(self, er_vocab, er_vocab_pairs, batch_size: int):
        np.random.shuffle(er_vocab_pairs)
        self.model.train()

        iters = len(er_vocab_pairs)
        with tqdm(total=iters, unit="ex", disable=not self.verbose, leave=False) as p:
            p.set_description("train loss")
            batch_start = 0

            while batch_start < iters:
                batch, targets = self.extract_batch(
                    er_vocab=er_vocab,
                    er_vocab_pairs=er_vocab_pairs,
                    batch_start=batch_start,
                    batch_size=batch_size,
                )
                l = self.step_on_batch(batch, targets)
                batch_start += batch_size
                p.update(batch_size)
                p.set_postfix(loss=f"{l.item():.2f}")

            if self.decay:
                self.scheduler.step()

    def step_on_batch(self, batch, targets):
        # if the batch has length 1 ( = this is the last batch) and the model has batch_norm layers,
        # do not try to update the batch_norm layers, because they would not work.
        if len(batch) == 1 and isinstance(self.model, ConvE):
            self.model.batch_norm_1.eval()
            self.model.batch_norm_2.eval()
            self.model.batch_norm_3.eval()

        self.optimizer.zero_grad()
        predictions = self.model.forward(batch)
        loss = self.loss(predictions, targets)
        loss.backward()
        self.optimizer.step()

        # if the layers had been set to mode "eval", put them back to mode "train"
        if len(batch) == 1 and isinstance(self.model, ConvE):
            self.model.batch_norm_1.train()
            self.model.batch_norm_2.train()
            self.model.batch_norm_3.train()

        return loss


class KelpieBCEOptimizer(BCEOptimizer):
    def __init__(self, model: Model, hp: dict, verbose: bool = True):
        super().__init__(model=model, hp=hp, verbose=verbose)

        self.optimizer = optim.Adam(params=self.model.parameters())

    def epoch(self, er_vocab, er_vocab_pairs, batch_size: int):
        self.model.train()

        with tqdm.tqdm(
            total=len(er_vocab_pairs), unit="ex", disable=not self.verbose
        ) as bar:
            bar.set_description("train loss")

            batch_start = 0
            while batch_start < len(er_vocab_pairs):
                batch, targets = self.extract_batch(
                    er_vocab=er_vocab,
                    er_vocab_pairs=er_vocab_pairs,
                    batch_start=batch_start,
                    batch_size=batch_size,
                )
                l = self.step_on_batch(batch, targets)

                self.model.update_embeddings()

                batch_start += batch_size
                bar.update(batch_size)
                bar.set_postfix(loss=str(round(l.item(), 6)))

            if self.decay:
                self.scheduler.step()

    def step_on_batch(self, batch, targets):
        if isinstance(self.model, ConvE):
            self.model.batch_norm_1.eval()
            self.model.batch_norm_2.eval()
            self.model.batch_norm_3.eval()
            self.model.convolutional_layer.eval()
            self.model.hidden_layer.eval()

        self.optimizer.zero_grad()
        predictions = self.model.forward(batch)
        loss = self.loss(predictions, targets)
        loss.backward()
        self.optimizer.step()

        return loss
