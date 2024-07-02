"""
    Reference-based attack, as described in https://proceedings.mlr.press/v97/sablayrolles19a/sablayrolles19a.pdf
"""

import numpy as np
import torch as ch

from mib.attacks.base import Attack


class Reference(Attack):
    """
    Reference-based attack
    """
    def __init__(self, model, criterion, device: str = "cuda", **kwargs):
        super().__init__(
            "Reference", model, criterion, device=device, reference_based=True
        )
        self.model.to(self.device)

    def compute_scores(self, x, y, **kwargs) -> np.ndarray:
        out_models = kwargs.get("out_models", None)
        if out_models is None:
            raise ValueError("Reference attack requires out_models to be specified")
        return compute_ref_score(self.model, x, y, self.device, out_models, self.criterion)


class ReferenceSmooth(Attack):
    """
    Reference-based attack that uses label-smoothing while computing loss
    """
    def __init__(self, model):
        super().__init__("ReferenceSmooth", model, reference_based=True, label_smoothing=0.05)

    def compute_scores(self, x, y, **kwargs) -> np.ndarray:
        out_models = kwargs.get("out_models", None)
        if out_models is None:
            raise ValueError("Reference attack requires out_models to be specified")
        return compute_ref_score(self.model, x, y, out_models, self.criterion)


class ReferenceAlex(Attack):
    """
    Reference-based attack, but aggregate scores in log-sum-exp manner as in Alex's paper
    """

    def __init__(self, model):
        super().__init__("ReferenceAlex", model, reference_based=True)

    def compute_scores(self, x, y, **kwargs) -> np.ndarray:
        out_models = kwargs.get("out_models", None)
        if out_models is None:
            raise ValueError("Reference attack requires out_models to be specified")
        return compute_ref_score(self.model, x, y, out_models, self.criterion, proper_ref_aggregate=True)


@ch.no_grad()
def compute_ref_score(model, x, y, device, out_models, criterion, proper_ref_aggregate: bool = False):
    x, y = x.to(device), y.to(device)
    logits = model(x).detach()
    if logits.shape[1] == 1:
        logits = logits.squeeze(1)
    loss = criterion(logits, y).cpu().numpy()

    ref_losses = []
    for out_model in out_models:
        out_model.to(device)
        logits = out_model(x).detach()
        if logits.shape[1] == 1:
            logits = logits.squeeze(1)
        ref_loss = (
            criterion(logits, y).cpu().numpy()
        )
        out_model.cpu()
        ref_losses.append(ref_loss)
    ref_losses = np.array(ref_losses)

    if proper_ref_aggregate:
        mean_out = np.log(np.mean(np.exp(-ref_losses), 0))
    else:
        mean_out = np.mean(ref_losses, 0)

    scores = np.array(loss - mean_out)
    return -scores
