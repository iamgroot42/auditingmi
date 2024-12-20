"""
    Finetuning/Unlearning based attacks.
"""
import numpy as np
import torch as ch
import torch.nn as nn
import copy

from mib.attacks.base import Attack
from mib.attacks.attack_utils import SpecificPointIncludedLoader, compute_gradients
from mib.train import train_model


class Unlearning(Attack):
    """
    Unlearning attack - records metrics before/after unlearning.
    """

    def __init__(self, model):
        super().__init__("Unlearning", model, whitebox=True)

    def compute_scores(self, x, y, **kwargs) -> np.ndarray:
        scores = []

        batch_size = kwargs.get("batch_size", 256)
        num_times = kwargs.get("num_times", 3)
        learning_rate = kwargs.get("learning_rate", 0.001)
        other_data_source = kwargs.get("other_data_source", None)
        if other_data_source is None:
            raise ValueError("other_data_source must be provided.")

        for x_, y_ in zip(x, y):
            # Create loader out of given data source
            given_loader = ch.utils.data.DataLoader(
                other_data_source, batch_size=batch_size - 1, shuffle=True
            )

            # Create custom loader
            loader = SpecificPointIncludedLoader(given_loader, (x_, y_), num_times)
            criterion = nn.CrossEntropyLoss()
            # Make copy of model before sending it over
            model_ = copy.deepcopy(self.model.cpu())

            # Take note of loss for interest_point before fine-tuning
            signals_before = self._measurement(
                model_,
                criterion,
                x_.unsqueeze(0),
                y_.unsqueeze(0),
            )

            # Fine-tune model
            model_.cuda()
            model_ = train_model(
                model_,
                criterion,
                loader,
                None,
                learning_rate=learning_rate,
                epochs=1,
                verbose=False,
                loss_multiplier=-1,
            )

            # Do sth with new model
            signals_after = self._measurement(
                model_,
                criterion,
                x_.unsqueeze(0),
                y_.unsqueeze(0),
            )

            # For now, look at difference in gradient norms
            score = np.linalg.norm(signals_after["grads"]) - np.linalg.norm(signals_before["grads"])
            # Closer to zero -> more likely member
            score = np.exp(-np.abs(score))
            scores.append(score)

        return np.array(scores)

    def _measurement(self, model, criterion, a, b):
        # Get model parameters
        params = []
        for p in model.parameters():
            params.extend(list(p.detach().cpu().numpy().flatten()))
        params = np.array(params)

        # Get model prediction (will use to compare later)
        pred_softmax = ch.nn.functional.softmax(model(a).detach().cpu(), dim=1).numpy()[
            0
        ]
        # Measure loss
        loss_measure = criterion(model(a), b).detach().cpu().item()

        # Compute gradient norm with given (a, b) datapoint
        grads = compute_gradients(model, criterion, a, b)

        return {
            "pred": pred_softmax,
            "loss": loss_measure,
            "grads": grads,
            "params": params,
        }


class UnlearningAct(Attack):
    """
    Unlearning attack - records activation metrics before/after unlearning.
    """

    def __init__(self, model):
        super().__init__("Unlearning", model, whitebox=True)

    def compute_scores(self, x, y, **kwargs) -> np.ndarray:
        scores = []

        batch_size = kwargs.get("batch_size", 256)
        num_times = kwargs.get("num_times", 3)
        learning_rate = kwargs.get("learning_rate", 0.001)
        other_data_source = kwargs.get("other_data_source", None)
        if other_data_source is None:
            raise ValueError("other_data_source must be provided.")

        # Take a reasonable amount of data from other_data_source
        other_data_stacked = ch.stack([z for z, _ in other_data_source], 0)[:batch_size]

        for x_, y_ in zip(x, y):
            # Create loader out of given data source
            given_loader = ch.utils.data.DataLoader(
                other_data_source, batch_size=batch_size - 1, shuffle=True
            )

            # Create custom loader
            loader = SpecificPointIncludedLoader(given_loader, (x_, y_), num_times)
            criterion = nn.CrossEntropyLoss()
            # Make copy of model before sending it over
            model_ = copy.deepcopy(self.model.cpu())

            # Take note of metric for interest_point before fine-tuning
            acts_before = self._measurement(
                model_,
                x_.unsqueeze(0),
            )
            # Also take note of metric for data definitely not seen before
            acts_before_out = self._measurement(model_, other_data_stacked)

            # Fine-tune model
            model_.cuda()
            model_ = train_model(
                model_,
                criterion,
                loader,
                test_loader=None,
                learning_rate=learning_rate,
                epochs=1,
                verbose=False,
                loss_multiplier=-1,
            )

            # Do sth with new model
            acts_after = self._measurement(
                model_,
                x_.unsqueeze(0),
            )
            # Also take note of metric for data definitely not seen before
            acts_after_out = self._measurement(model_, other_data_stacked)

            # For now, look at difference in gradient norms
            score = acts_after - acts_before
            # Normalize by difference in activation for out-of-distribution data
            score -= (acts_after_out - acts_before_out)
            scores.append(score)

        return np.array(scores)

    def _measurement(self, model, a):
        pick_layer = 2 # doesn't seem to matter much

        # With ref (non0member data)
        # 2 : 0.528
        # 3 : 0.529

        acts = (
            model(a, layer_readout=pick_layer)
            .detach()
            .cpu()
            .view(len(a), -1)
            .numpy()
        )

        nonzero_acts = np.mean(np.sum(acts > 0, 1) * 1.)
        return nonzero_acts
