"""
    Implementation for LOSS attack and some variants.
"""
import torch as ch
import numpy as np

from mib.attacks.base import Attack
from mib.attacks.attack_utils import compute_scaled_logit


class LOSS(Attack):
    """
    Standard LOSS attack.
    """

    def __init__(self, model, criterion, device: str = "cuda", **kwargs):
        super().__init__("LOSS", model, criterion, device=device)
        self.model.to(self.device)

    @ch.no_grad()
    def compute_scores(self, x, y, **kwargs) -> np.ndarray:
        logits = self.model(x.to(self.device)).detach()
        if logits.shape[1] == 1:
            logits = logits.squeeze(1)
        loss = self.criterion(logits, y.to(self.device))
        return -loss.cpu().numpy()


class LOSSSmooth(Attack):
    """
    Standard LOSS attack that uses label-smoothing while computing loss.
    """

    def __init__(self, model, criterion, device: str = "cuda", **kwargs):
        # 0 (no smoothing) -> 0.615 (random low-FPR)
        # 1e-2 -> 0.547
        # 5e-3 -> 0.566
        # 1e-3 -> 0.616
        # 5e-4 -> 0.628 (decent low-FPR)
        # 1e-4 -> 0.630 (bad low-FPR)
        super().__init__("LOSSSmooth", model, criterion, device=device, label_smoothing=5e-4)
        self.model.to(self.device)

    @ch.no_grad()
    def compute_scores(self, x, y, **kwargs) -> np.ndarray:
        loss = self.criterion(self.model(x.to(self.device)).detach(), y.to(self.device))
        return -loss.cpu().numpy()


class Logit(Attack):
    """
    Logit-scaling applied to model confidence
    """

    def __init__(self, model, criterion, device: str = "cuda", **kwargs):
        super().__init__("LOSS_LiRA", model, criterion, device=device)
        self.model.to(self.device)

    @ch.no_grad()
    def compute_scores(self, x, y, **kwargs) -> np.ndarray:
        return compute_scaled_logit(self.model, x.to(self.device), y.to(self.device))
