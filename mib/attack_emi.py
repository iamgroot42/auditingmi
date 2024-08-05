"""
    File to run L-attack using 'Enhanced Membership Infernece' work
    Computes threshold at given FPR using reference models, instead of direct calibration.
"""
import argparse
import os
import torch as ch
from scipy.stats import norm as Normal
import numpy as np
from tqdm import tqdm
from typing import List

from mib.models.utils import get_model
from sklearn.metrics import auc
from mib.dataset.utils import get_dataset
from mib.attacks.utils import get_attack
from mib.utils import (
    get_models_path,
    load_ref_models,
)
from mib.train import get_loader
from mib.attack import member_nonmember_loaders


def get_signals(
    args,
    loader,
    ds,
    target_model,
    criterion,
    is_train: bool,
    model_dir: str
):

    # Get LOSS attacker
    target_loss = get_attack("LOSS")(target_model, criterion, device="cuda")

    losses_record, losses_ref = [], []

    # Compute signals for member data
    for i, (x, y) in tqdm(enumerate(loader), total=len(loader), desc="Member" if is_train else "Non-member"):

        # Collect loss for target model
        loss_t = - target_loss.compute_scores(x, y)
        losses_record.append(loss_t)

        ref_loss_fns_use = None
        if is_train:
            # If is_train, load up reference models
            this_dir = os.path.join(
                        model_dir, f"l_mode_in/{args.target_model_index}/{i}"
                    )
            out_models_use, _ = load_ref_models(this_dir, args, ds.num_classes)
            ref_loss_fns_use = [get_attack("LOSS")(ref_model, criterion, device="cuda") for ref_model in out_models_use]

            losses_ref.append([ -ref_loss_fn.compute_scores(x, y) for ref_loss_fn in ref_loss_fns_use])

    losses_record = np.array(losses_record)
    if not is_train:
        return losses_record

    losses_ref = np.array(losses_ref)
    return losses_record, losses_ref


def get_threshold_linear(scores, fp_rate):
    return np.quantile(scores, fp_rate, method="linear")


def get_threshold_gaussian(scores, fp_rate):
    logit_scaling = lambda x: np.log(np.exp(-x) / (1 - np.exp(-x)))
    logit_scores = logit_scaling(scores)

    mean, std = np.mean(logit_scores), np.std(logit_scores)
    # Define Normal distribution
    percentile = Normal.ppf(
        1 - fp_rate,
        loc=mean,
        scale=std,
    )

    # Reverse the logit rescaling
    threshold = -np.log(1 / (1 + np.exp(-percentile)))
    return threshold


def main(args):
    model_dir = os.path.join(
        get_models_path(),
        args.dataset,
        args.model_arch,
        f"lr_{args.momentum}_wd_{args.weight_decay}",
    )

    ds = get_dataset(args.dataset)(augment=False)

    # Load target model
    target_model, criterion, _ = get_model(args.model_arch, ds.num_classes)
    model_dict = ch.load(
        os.path.join(model_dir, f"{args.target_model_index}.pt"), map_location="cpu"
    )
    target_model.load_state_dict(model_dict["model"], strict=False)
    target_model.eval()

    # Pick records (out of all train) to test
    train_index = model_dict["train_index"]

    # TODO: Get these stats from a dataset class
    num_nontrain_pool = 10000

    # Get data
    train_data = ds.get_train_data()

    if args.partial_model is not None:
        # Skip train_data with corresponding features
        # using target_model(layer_readout=args.partial_model)
        # do it in batches, of course
        X_, Y_ = [], []
        target_model.cuda()
        with ch.no_grad():
            loader = get_loader(train_data, None, 128, num_workers=0)
            for x, y in tqdm(
                loader, desc=f"Collecting features from layer {args.partial_model}"
            ):
                features = (
                    target_model(x.cuda(), layer_readout=args.partial_model)
                    .cpu()
                    .detach()
                )
                X_.append(features)
                Y_.append(y)
        # Make a new dataset out of X_ and Y_
        target_model.cpu()
        train_data = ch.utils.data.TensorDataset(ch.cat(X_), ch.cat(Y_))

    # Get data split
    (
        member_loader,
        nonmember_loader,
        _,
        _,
        _,
    ) = member_nonmember_loaders(
        train_data,
        train_index,
        args.num_points,
        args.exp_seed,
        num_nontrain_pool=num_nontrain_pool,
        split_each_loader=1,
        l_mode=True,
    )

    # Extract "later" part of model
    if args.partial_model is not None:
        model_for_ihvp = target_model.make_later_layer_model(args.partial_model + 1)
        model_for_ihvp.eval()
    else:
        model_for_ihvp = target_model

    losses_mem, losses_ref_mem = get_signals(
        args, member_loader, 
        ds, model_for_ihvp,
        criterion, is_train=True,
        model_dir=model_dir)
    losses_nonmem = get_signals(
        args, nonmember_loader, 
        ds, model_for_ihvp,
        criterion, is_train=False,
        model_dir=model_dir)

    # Assert length of signals is same
    if len(losses_mem) != len(losses_nonmem):
        print(
            f"Length of losses_mem and losses_nonmem don't match. Found {len(losses_mem)} and {len(losses_nonmem)} respectively."
        )
        exit(0)

    tprs, fprs = [], []
    for fpr in np.linspace(0, 1.0, 1001):
        fprs.append(fpr)
        # Look at histogram of losses_ref_mem for each record and get threshold that provides that FPR
        thresholds = np.array([get_threshold_linear(losses_ref_mem[i], fpr) for i in range(len(losses_ref_mem))])
        # Compute TPR with these thresholds
        tpr = (losses_mem <= thresholds).mean()
        tprs.append(tpr)

    # Compute AUC
    roc_auc = auc(fprs, tprs)

    # Print out ROC
    print(f"ROC AUC: {roc_auc}")
    # Also print out TPR at 1% and 0.1% FPR
    print(f"TPR at 1% FPR: {tprs[fprs.index(0.01)]}")
    print(f"TPR at 0.1% FPR: {tprs[fprs.index(0.001)]}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    # Attack setup
    args.add_argument(
        "--target_model_index", type=int, default=0, help="Index of target model"
    )
    args.add_argument("--num_ref_models", type=int, default=None)
    args.add_argument("--attack", type=str, default="LOSS")
    args.add_argument("--exp_seed", type=int, default=2024)
    args.add_argument("--aug", action="store_true", help="Use augmented data?")
    args.add_argument(
        "--num_points",
        type=int,
        default=500,
        help="Number of samples (in and out each) to use for computing signals",
    )
    args.add_argument(
        "--same_seed_ref",
        action="store_true",
        help="Use ref models with same seed as target model?",
    )
    args.add_argument(
        "--specific_ref_folder",
        type=str,
        default=None,
        help="Custom ref sub-folder to load ref models from",
    )
    args.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Custom suffix (folder) to load models from",
    )
    # Dataset/model related
    args.add_argument("--model_arch", type=str, default="wide_resnet_28_2")
    args.add_argument("--dataset", type=str, default="cifar10")
    args.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum for SGD optimizer."
    )
    args.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay for SGD optimizer.",
    )
    args.add_argument(
        "--partial_model",
        type=int,
        default=None,
        help="Part of model from this specified layer is treated as actual model. Helps in computation for Hessian-based attacks",
    )
    args.add_argument(
        "--simulate_metaclf",
        action="store_true",
        help="If true, use scores as features and fit LOO-style meta-classifier for each target datapoint",
    )
    args.add_argument(
        "--skip_save",
        action="store_true",
        help="If true, so not save scores to file",
    )

    args.add_argument(
        "--sif_proper_mode",
        action="store_true",
        help="Tune 2 thresholds for SIF like original paper",
    )
    args = args.parse_args()

    main(args)
