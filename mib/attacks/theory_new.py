"""
    Train a m-layer MLP with multivariate Gaussian distribution as data.
    Allows to work in simplistic setting and have potentially closed-form solutions for these attacks.
"""

import numpy as np
import torch as ch

from mib.attacks.base import Attack
from torch.autograd import grad
from torch_influence import BaseObjective
from torch_influence import AutogradInfluenceModule, LiSSAInfluenceModule, HVPModule, CGInfluenceModule


class MyObjective(BaseObjective):
    def __init__(self, criterion, weight_decay: float = 0):
        self._criterion = criterion
        self._weight_decay = weight_decay

    def train_outputs(self, model, batch):
        return model(batch[0])

    def train_loss_on_outputs(self, outputs, batch):
        if outputs.shape[1] == 1:
            return self._criterion(
                outputs.squeeze(1), batch[1].float()
            )  # mean reduction required
        else:
            return self._criterion(outputs, batch[1])  # mean reduction required

    def train_regularization(self, params):
        if self._weight_decay == 0:
            return 0
        return 0.5 * self._weight_decay * ch.square(params.norm())

    def test_loss(self, model, params, batch):
        output = model(batch[0])
        # no regularization in test loss
        if output.shape[1] == 1:
            return self._criterion(output.squeeze(1), batch[1].float())
        else:
            return self._criterion(output, batch[1])


class IHA(Attack):
    """
    I1 + I3 based term that makes assumption about relationship between I2 and I3.
    """
    def __init__(self, model, criterion, device: str = "cuda", **kwargs):
        all_train_loader = kwargs.get("all_train_loader", None)
        approximate = kwargs.get("approximate", False) # Use approximate iHVP instead of exact?
        hessian = kwargs.get("hessian", None) # Precomputed Hessian, if available
        damping_eps = kwargs.get("damping_eps", 2e-1) # Damping (or cutoff for low-rank approximation)
        low_rank = kwargs.get("low_rank", False) # Use low-rank approximation for Hessian?
        tol = kwargs.get("tol", 1e-5) # Tolerance for CG solver (if approximate is True)
        self.weight_decay = kwargs.get("weight_decay", 5e-4) # Weight decay used to train model

        if all_train_loader is None:
            raise ValueError("IHA requires all_train_loader to be specified")
        super().__init__(
            "IHA",
            model,
            criterion,
            device=device,
            reference_based=False,
            requires_trace=False,
            whitebox=True,
            uses_hessian=not approximate,
        )

        self.approximate = approximate
        self.model.to(self.device, non_blocking=True)

        self.all_data_grad = self.collect_grad_on_all_data(all_train_loader)

        # Exact Hessian
        if self.approximate:
            # LiSSA - 40s/iteration
            # GC - 38s/iteration
            """
            self.ihvp_module = LiSSAInfluenceModule(
                model=model,
                objective=MyObjective(criterion),
                train_loader=all_train_loader,
                test_loader=None,
                device=self.device,
                repeat=4,
                depth=100,  # 5000 for MLP and Transformer, 10000 for CNN
                scale=25,
                damp=damping_eps,
            )
            """
            self.ihvp_module = CGInfluenceModule(
                model=model,
                objective=MyObjective(criterion),
                train_loader=all_train_loader,
                test_loader=None,
                device=self.device,
                damp=damping_eps,
                tol=tol,
            )
        else:
            if hessian is None:
                exact_H = compute_hessian(
                    model,
                    all_train_loader,
                    self.criterion,
                    weight_decay=self.weight_decay,
                    device=self.device,
                    verbose=True,
                )
                self.hessian = exact_H.cpu().clone().detach()
            else:
                self.hessian = hessian

            L, Q = ch.linalg.eigh(self.hessian)

            if low_rank:
                # Low-rank approximation
                qualifying_indices = ch.abs(L) > damping_eps
                Q_select = Q[:, qualifying_indices]
                self.H_inverse = Q_select @ ch.diag(1 / L[qualifying_indices]) @ Q_select.T
            else:
                # Damping
                L += damping_eps
                self.H_inverse = Q @ ch.diag(1 / L) @ Q.T

    def collect_grad_on_all_data(self, loader):
        cumulative_gradients = None
        for x, y in loader:
            # Zero-out accumulation
            self.model.zero_grad()
            # Compute gradients
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            if logits.shape[1] == 1:
                loss = self.criterion(logits.squeeze(), y.float()) * len(x)
            else:
                loss = self.criterion(logits, y) * len(x)
            loss.backward()
            flat_grad = []
            for p in self.model.parameters():
                if p.grad is not None:
                    flat_grad.append(p.grad.detach().view(-1))
            # Flatten out gradients
            flat_grad = ch.cat(flat_grad)
            # Accumulate in higher precision
            if cumulative_gradients is None:
                cumulative_gradients = ch.zeros_like(flat_grad)
            cumulative_gradients += flat_grad
        self.model.zero_grad()
        cumulative_gradients /= len(loader.dataset)
        return cumulative_gradients

    def get_specific_grad(self, point_x, point_y):
        self.model.zero_grad()
        logits = self.model(point_x.to(self.device))
        if logits.shape[1] == 1:
            loss = self.criterion(logits.squeeze(1), point_y.float().to(self.device))
        else:
            loss = self.criterion(logits, point_y.to(self.device))
        ret_loss = loss.item()
        loss.backward()
        flat_grad = []
        for p in self.model.parameters():
            flat_grad.append(p.grad.detach().view(-1))
        flat_grad = ch.cat(flat_grad)
        self.model.zero_grad()
        return flat_grad, ret_loss

    def compute_scores(self, x, y, **kwargs) -> np.ndarray:
        x, y = x.to(self.device), y.to(self.device)
        learning_rate = kwargs.get("learning_rate", None) # Learning rate used to train model
        num_samples = kwargs.get("num_samples", None)
        is_train = kwargs.get("is_train", None)
        momentum = kwargs.get("momentum", 0.9) # Momentum used to train model

        skip_reg_term = kwargs.get("skip_reg_term", False)  # Skip the extra-computation regularization term?
        skip_loss = kwargs.get("skip_loss", False)  # Skip loss, directly use (I1 + I2 + I3 + I4)?
        get_individual_terms = kwargs.get("get_individual_terms", False) # Get terms I1-I4 as well?
        only_i1 = kwargs.get("only_i1", False) # Only compute I1 term?
        only_i2 = kwargs.get("only_i2", False)  # Only compute I2 term?
        i1i2_include_loss = kwargs.get("i1i2_include_loss", False) # When using only_i1 or only_i2, include loss?

        if is_train is None:
            raise ValueError(f"{self.name} requires is_train to be specified (to compute L0 properly)")
        if learning_rate is None:
            raise ValueError(f"{self.name} requires knowledge of learning_rate")
        if num_samples is None:
            raise ValueError(f"{self.name} requires knowledge of num_samples")

        # Factor out S/(2L*) parts out of both terms as common
        grad, ret_loss = self.get_specific_grad(x, y)

        if skip_loss:
            I1 = 0
        else:
            I1 = ret_loss / (1 + momentum)

        if is_train:
            # Trick to skip computing L0 for all records. Compute L1 (across all data), and then directly calculate L0 using current grad
            # We are passing 'is_train' flag here but this is not cheating- could be replaced with repeated L0 computation, but would be unnecessarily expensive
            # all_other_data_grad = self.all_data_grad - (grad / num_samples)
            all_other_data_grad = (self.all_data_grad * num_samples - grad) / (num_samples - 1)
        else:
            all_other_data_grad = self.all_data_grad

        if self.approximate:
            # H-1 * grad(l(z))
            datapoint_ihvp = self.ihvp_module.inverse_hvp(grad)
            # H-1 * grad(L0(z))
            ihvp_alldata   = self.ihvp_module.inverse_hvp(all_other_data_grad)

            if self.weight_decay > 0 and not skip_reg_term:
                extra_ihvp = self.ihvp_module.inverse_hvp(datapoint_ihvp)
        else:
            # H-1 * grad(l(z))
            datapoint_ihvp = (self.H_inverse @ grad.cpu()).to(self.device, non_blocking=True)
            # H-1 * grad(L0(z))
            ihvp_alldata   = (self.H_inverse @ all_other_data_grad.cpu()).to(self.device)
            if self.weight_decay > 0 and not skip_reg_term:
                extra_ihvp = (self.H_inverse @ datapoint_ihvp.cpu()).to(self.device, non_blocking=True)

        I2 = ch.dot(datapoint_ihvp, datapoint_ihvp).cpu().item() / num_samples
        I3 = ch.dot(ihvp_alldata, datapoint_ihvp).cpu().item() * 2

        extra_reg_term = 0
        if self.weight_decay > 0 and not skip_reg_term:
            I4 = ch.dot(datapoint_ihvp, extra_ihvp).cpu().item() / num_samples
            I5 = ch.dot(ihvp_alldata, extra_ihvp).cpu().item() * 2
            # Old (unnecessary approximation)
            # multiplier = self.weight_decay * 2
            # Correct
            multiplier = (self.weight_decay / 2) * (2 - ((learning_rate * self.weight_decay) / (1 + momentum)))
            extra_reg_term = -(I4 + I5) * multiplier

        if self.weight_decay > 0:
            scaling = 1 - ((learning_rate * self.weight_decay) / (1 + momentum))
        else:
            scaling = 1

        if not skip_loss:
            scaling /= learning_rate
            extra_reg_term /= learning_rate

        if only_i1:
            if i1i2_include_loss:
                mi_score = I1 - (I2 * scaling)
            else:
                mi_score = - I2
        elif only_i2:
            if i1i2_include_loss:
                mi_score = I1 - (I3 * scaling)
            else:
                mi_score = - I3
        else:
            mi_score = I1 - ((I2 + I3) * scaling) + extra_reg_term

        # Return individual scores, if requested
        if get_individual_terms:
            return_dict = {
                "I2": I2 * scaling,
                "I3": I3 * scaling,
            }
            if not skip_loss:
                return_dict["I1"] = I1
            if self.weight_decay > 0 and not skip_reg_term:
                return_dict["I4"] = I4 * self.weight_decay / learning_rate
                return_dict["I5"] = I5 * self.weight_decay / learning_rate

            return return_dict

        return mi_score


def compute_hessian(
    model, loader, criterion, weight_decay,
    device: str = "cpu", verbose: bool = False
):
    """
    Compute Hessian at given point
    """
    model.zero_grad()

    module = AutogradInfluenceModule(
        model=model,
        objective=MyObjective(criterion),
        train_loader=loader,
        test_loader=None,
        device=device,
        damp=0,
        store_as_hessian=True,
        verbose=verbose,
    )

    H = module.get_hessian()
    return H


def compute_epsilon_acceleration(
    source_sequence,
    num_applications: int = 1,
):
    """Compute `num_applications` recursive Shanks transformation of
    `source_sequence` (preferring later elements) using `Samelson` inverse and the
    epsilon-algorithm, with Sablonniere modifier.
    """

    def inverse(vector):
        # Samelson inverse
        return vector / vector.dot(vector)

    epsilon = {}
    for m, source_m in enumerate(source_sequence):
        epsilon[m, 0] = source_m
        epsilon[m + 1, -1] = 0

    s = 1
    m = (len(source_sequence) - 1) - 2 * num_applications
    initial_m = m
    while m < len(source_sequence) - 1:
        while m >= initial_m:
            # Sablonniere modifier
            inverse_scaling = np.floor(s / 2) + 1

            epsilon[m, s] = epsilon[m + 1, s - 2] + inverse_scaling * inverse(
                epsilon[m + 1, s - 1] - epsilon[m, s - 1]
            )
            epsilon.pop((m + 1, s - 2))
            m -= 1
            s += 1
        m += 1
        s -= 1
        epsilon.pop((m, s - 1))
        m = initial_m + s
        s = 1

    return epsilon[initial_m, 2 * num_applications]


@ch.no_grad()
def hso_ihvp(
    vec,
    hvp_module,
    acceleration_order: int = 9,
    initial_scale_factor: float = 1e6,
    num_update_steps: int = 20,
):

    # Detach and clone input
    vector_cache = vec.detach().clone()
    update_sum = vec.detach().clone()
    coefficient_cache = 1

    cached_update_sums = []
    if acceleration_order > 0 and num_update_steps == 2 * acceleration_order + 1:
        cached_update_sums.append(update_sum)

    # Do HessianSeries calculation
    for update_step in range(1, num_update_steps):
        hessian2_vector_cache = hvp_module.hvp(hvp_module.hvp(vector_cache))

        if update_step == 1:
            scale_factor = ch.norm(hessian2_vector_cache, p=2) / ch.norm(vec, p=2)
            scale_factor = max(scale_factor.item(), initial_scale_factor)

        vector_cache = vector_cache - (1 / scale_factor) * hessian2_vector_cache
        coefficient_cache *= (2 * update_step - 1) / (2 * update_step)
        update_sum += coefficient_cache * vector_cache

        if acceleration_order > 0 and update_step >= (
            num_update_steps - 2 * acceleration_order - 1
        ):
            cached_update_sums.append(update_sum.clone())

    update_sum /= np.sqrt(scale_factor)

    # Perform series acceleration (Shanks acceleration)
    if acceleration_order > 0:
        accelerated_sum = compute_epsilon_acceleration(
            cached_update_sums, num_applications=acceleration_order
        )
        accelerated_sum /= np.sqrt(scale_factor)
        return accelerated_sum

    return update_sum


if __name__ == "__main__":
    import torch.nn as nn
    m = nn.Sequential(
        nn.Linear(600, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )
    device = "cuda"
    m.to(device)
    x = ch.rand(10, 600)
    y = ch.tensor([1, 0, 1, 0, 1, 0, 1, 1, 0, 1]).unsqueeze(1).float()
    criterion = nn.BCEWithLogitsLoss()
    # Make proper pytorch loader out of (x, y)
    loader = ch.utils.data.DataLoader(ch.utils.data.TensorDataset(x, y), batch_size=3)

    # Compute and get grads for model
    loss = criterion(m(x.to(device)), y.to(device))
    loss.backward()
    flat_grad = []
    for p in m.parameters():
        if p.grad is not None:
            flat_grad.append(p.grad.detach().view(-1))
    flat_grad = ch.cat(flat_grad)
    m.zero_grad()

    # Get exact Hessian
    H = compute_hessian(m, loader, criterion, weight_decay=0, device=device)
    print(H)
