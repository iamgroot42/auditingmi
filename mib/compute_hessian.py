"""
    Generating and storing Hessian for a target model.
"""

import argparse
import os
import torch as ch

from mib.models.utils import get_model
from mib.dataset.utils import get_dataset
from mib.attacks.utils import get_attack
from mib.utils import get_models_path, get_misc_path
from mib.train import get_loader

"""
# Deterministic
ch.use_deterministic_algorithms(True)
ch.backends.cudnn.deterministic = True
ch.backends.cudnn.benchmark = False
"""

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

    # Get data
    train_data = ds.get_train_data()

    hessian = None
    # If attack uses uses_hessian, try loading from disk if available
    hessian_store_path = os.path.join(
        get_misc_path(),
        args.dataset,
        f"{args.model_arch}_lr_{args.momentum}_wd_{args.weight_decay}",
        str(args.target_model_index),
    )
    if os.path.exists(os.path.join(hessian_store_path, "hessian.ch")):
        hessian = ch.load(os.path.join(hessian_store_path, "hessian.ch"))
        print("Loaded Hessian!")

    attacker_obj = get_attack("IHA")(
        target_model,
        criterion,
        all_train_loader=get_loader(
            train_data, train_index, args.hessian_batch_size, num_workers=4
        ),
        hessian=hessian,
        damping_eps=2e-1,
        low_rank=False,
        save_compute_trick=False,
        approximate=False,
        tol=1e-5,
        device=f"cuda:0",
    )

    # Save Hessian, if computed and didn't exist before
    if attacker_obj.uses_hessian and (
        not os.path.exists(os.path.join(hessian_store_path, "hessian.ch"))
    ):
        os.makedirs(hessian_store_path, exist_ok=True)
        ch.save(
            attacker_obj.get_hessian(),
            os.path.join(hessian_store_path, "hessian.ch"),
        )
        print("Saved Hessian!")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_arch", type=str, default="wide_resnet_28_2")
    args.add_argument("--dataset", type=str, default="cifar10")
    args.add_argument(
        "--hessian_batch_size",
        type=int,
        default=256,
        help="Batch size for Hessian computation",
    )
    args.add_argument("--exp_seed", type=int, default=2024)
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
        "--target_model_index", type=int, default=0, help="Index of target model"
    )
    args.add_argument("--num_ref_models", type=int, default=None)
    args.add_argument(
        "--simulate_metaclf",
        action="store_true",
        help="If true, use scores as features and fit LOO-style meta-classifier for each target datapoint",
    )
    args.add_argument("--aug", action="store_true", help="Use augmented data?")
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
    args = args.parse_args()
    main(args)
