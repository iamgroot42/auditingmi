"""
    Analysis of outliers and how well the Hessian assumption (H0 ~ H1) holds for these points,
    compared to other datapoints.
"""
import argparse
import os
import copy
import torch as ch
import numpy as np
from tqdm import tqdm

from mib.models.utils import get_model
from mib.dataset.utils import get_dataset
from mib.utils import get_models_path
from mib.train import get_loader, train_model
from mib.attacks.theory_new import compute_hessian


def train_models_multiple(
    args,
    train_index,
    train_data,
    model_init,
    test_index,
    test_data,
    ds,
    batch_size: int,
    device: str,
    filename: str,
):
    """
        Train model on specified data multiple times, save Hessian for each case
    """
    # Make loaders
    train_loader = get_loader(train_data, train_index, batch_size)
    test_loader  = get_loader(test_data, test_index, batch_size)

    hessians = []
    for _ in range(args.n_models):
        model, criterion, hparams = get_model(args.model_arch, ds.num_classes)
        model.load_state_dict(model_init)
        # model.load_state_dict(copy.deepcopy(model_init))
        model.to(device)

        # Train model
        model = train_model(
            model,
            criterion,
            train_loader,
            test_loader,
            hparams["learning_rate"],
            hparams["epochs"],
            device=device,
        )[0]

        # Compute Hessian for this model
        exact_H = compute_hessian(model, train_loader, criterion, device=device)
        hessian = exact_H.cpu().clone().detach().numpy()
        hessians.append(hessian)

    # Save hessians
    hessians = np.stack(hessians, axis=0)

    # Dump Hessian
    print("Saving Hessians to???", filename)
    np.save(filename, hessians)


def main(args):
    # Open information related to outlier points
    model_dir = os.path.join(get_models_path(), args.dataset, args.model_arch, f"lr_{args.momentum}_wd_{args.weight_decay}")
    ds = get_dataset(args.dataset)(augment=False)
    batch_size = 128

    # Load target model
    model_dict = ch.load(os.path.join(model_dir, f"{args.target_model_index}.pt"), map_location="cpu")
    # Get model_init from this
    model_init = model_dict["model_init"]
    test_index = model_dict["test_index"]

    # Pick records (out of all train) to test
    train_index = model_dict["train_index"]

    # Extract actual data
    train_data = ds.train
    test_data = ds.test

    # Now load up indices for outliers (.npz)
    path = f"./ihp_study/outliers_{args.dataset}_{args.model_arch}_{args.target_model_index}.npz"
    with np.load(path) as data:
        low_diff = data["low_diff"]
        high_diff = data["high_diff"]

    # Get corresponding data indices for low_diff and high_diff
    low_diff_full = train_index[low_diff]
    high_diff_full = train_index[high_diff]
    everything_else = np.setdiff1d(train_index, np.concatenate([low_diff, high_diff]))

    # Sample args.n_sample points from each of these categories
    np.random.seed(args.exp_seed + 99)
    low_diff = np.random.choice(low_diff_full, args.n_sample, replace=False)
    high_diff = np.random.choice(high_diff_full, args.n_sample, replace=False)
    everything_else = np.random.choice(everything_else, args.n_sample, replace=False)

    # For clean data, pick all indices \ {low_diff (full), high_diff(full), and everything_else (picked)}
    indices_sample_from = np.setdiff1d(
        np.arange(len(train_data)),
        np.concatenate([low_diff_full, high_diff_full, everything_else]),
    )

    # Create samples of {D}
    D_indices_samples = []
    for i in range(args.n_clean):
        # Sample points from the indices_sample_from
        np.random.seed(args.exp_seed + 100 + i)
        D_indices_samples.append(np.random.choice(indices_sample_from, len(train_index), replace=False))

    # Make sure relevant directories exist
    hessians_save_path = f"./ihp_study/hessians/{args.dataset}_{args.model_arch}_{args.target_model_index}/"
    for make_sure_exist in ["clean", "low_diff", "high_diff", "everything_else"]:
        os.makedirs(os.path.join(hessians_save_path, make_sure_exist), exist_ok=True)

    # Train clean models in parallel
    # Hessian computation can get expensive, so only one to run one process per GPU
    for i, D_sample in tqdm(enumerate(D_indices_samples), desc="Clean models"):
        filename = os.path.join(hessians_save_path, "clean", f"{i}.npy")

        # Train on D_sample
        train_models_multiple(args,
                D_sample,
                train_data,
                copy.deepcopy(model_init),
                test_index,
                test_data,
                ds,
                batch_size,
                "cuda",
                filename
        )

    # For each clean model
    # For this, we assume at least 3 GPUs
    for j, D_sample in enumerate(D_indices_samples):

        # Make sure directories exist
        os.makedirs(os.path.join(hessians_save_path, "low_diff", f"{j}"), exist_ok=True)
        os.makedirs(os.path.join(hessians_save_path, "high_diff", f"{j}"), exist_ok=True)
        os.makedirs(os.path.join(hessians_save_path, "everything_else", f"{j}"), exist_ok=True)

        # Now also run processes for low_diff, high_diff, and everything_else
        # One point at a time from all three
        for i in tqdm(range(args.n_sample), "Outliers"):
            # Low-diff outlier
            train_models_multiple(args,
                    np.concatenate([D_sample, [low_diff[i]]]),
                    train_data,
                    copy.deepcopy(model_init),
                    test_index,
                    test_data,
                    ds,
                    batch_size,
                    "cuda",
                    os.path.join(hessians_save_path, "low_diff", f"{j}", f"{i}.npy")
            )

            # High-diff outlier
            train_models_multiple(args,
                    np.concatenate([D_sample, [high_diff[i]]]),
                    train_data,
                    copy.deepcopy(model_init),
                    test_index,
                    test_data,
                    ds,
                    batch_size,
                    "cuda",
                    os.path.join(hessians_save_path, "high_diff", f"{j}", f"{i}.npy")
            )

            # Everything else
            train_models_multiple(args,
                    np.concatenate([D_sample, [everything_else[i]]]),
                    train_data,
                    copy.deepcopy(model_init),
                    test_index,
                    test_data,
                    ds,
                    batch_size,
                    "cuda",
                    os.path.join(hessians_save_path, "everything_else", f"{j}", f"{i}.npy")
            )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_arch", type=str, default="wide_resnet_28_2")
    args.add_argument("--dataset", type=str, default="cifar10")
    args.add_argument("--target_model_index", type=int, default=0)
    args.add_argument("--n_sample", type=int, default=100, help="Number of samples to draw from the outlier/inlier selection for H0/H1 study")
    args.add_argument("--n_clean", type=int, default=10, help="Number of times D should be resampled")
    args.add_argument("--n_models", type=int, default=5, help="Number of models (per subset) to train for H0/H1 study")
    args.add_argument("--exp_seed", type=int, default=2024)
    args.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer.")
    args.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay for SGD optimizer.")
    args = args.parse_args()

    save_dir = get_models_path()
    main(args)
