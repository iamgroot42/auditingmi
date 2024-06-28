"""
    File for generating signal corresponding to a given attack.
    Scores can be read later from the saved file.
"""
import argparse
import os
import copy
import torch as ch
import numpy as np
from tqdm import tqdm
import multiprocessing
import torch.multiprocessing as mp

from mib.models.utils import get_model
from sklearn.metrics import roc_curve, auc
from mib.dataset.utils import get_dataset
from mib.attacks.utils import get_attack
from mib.utils import get_signals_path, get_models_path, get_misc_path, DillProcess
from mib.train import get_loader
from sklearn.ensemble import RandomForestClassifier

"""
# Deterministic
ch.use_deterministic_algorithms(True)
ch.backends.cudnn.deterministic = True
ch.backends.cudnn.benchmark = False
"""

def member_nonmember_loaders(
    train_data,
    train_idx,
    num_points_sample: int,
    exp_seed: int,
    num_nontrain_pool: int = None,
    batch_size: int = 1,
    want_all_member_nonmember: bool = False,
    split_each_loader: int = 1
):
    """
    num_points_sample = -1 means all points should be used, and # of non-members will be set to be = # of members
    """
    
    other_indices_train = np.array(
        [i for i in range(len(train_data)) if i not in train_idx]
    )

    want_all_members_and_equal_non_members = num_points_sample == -1
    if want_all_members_and_equal_non_members:
        num_points_sample = min(len(train_idx), len(other_indices_train))

    # Create Subset datasets for members
    if want_all_member_nonmember:
        train_index_subset = train_idx
    else:
        np.random.seed(exp_seed)
        train_index_subset = np.random.choice(train_idx, num_points_sample, replace=False)

    # Sample non-members
    np.random.seed(exp_seed + 1)
    nonmember_indices = np.random.choice(
        other_indices_train,
        (
            num_nontrain_pool
            if not want_all_members_and_equal_non_members
            else num_points_sample
        ),
        replace=False,
    )

    if want_all_member_nonmember or want_all_members_and_equal_non_members:
        nonmember_index_subset = nonmember_indices
        nonmember_dset_ft = None
    else:
        # Break nonmember_indices here into 2 - one for sprinkling in FT data, other for actual non-members
        nonmember_indices_ft = nonmember_indices[: num_nontrain_pool // 2]
        nonmember_indices_test = nonmember_indices[num_nontrain_pool // 2 :]

        nonmember_dset_ft = ch.utils.data.Subset(train_data, nonmember_indices_ft)

        # Sample non-members
        np.random.seed(exp_seed + 2)
        nonmember_index_subset = np.random.choice(
            nonmember_indices_test,
            num_points_sample,
            replace=False,
        )

    # Assert no overlap between train_index_subset and nonmember_index_subset
    # Just to make sure nothing went wrong above!
    if len(set(train_index_subset).intersection(set(nonmember_index_subset))) != 0:
        print("Non-overlap found between train and non-member data. Shouldn't have happened! Check code.")
        exit(0)

    # Make dsets
    if split_each_loader > 1:
        train_index_splits = np.array_split(train_index_subset, split_each_loader)
        nonmember_index_splits = np.array_split(nonmember_index_subset, split_each_loader)

        train_index_subset_, nonmember_index_subset_ = [], []
        member_loader, nonmember_loader = [], []
        for (mem_split, nonmem_split) in zip(train_index_splits, nonmember_index_splits):
            # Save split information (for model mapping)
            train_index_subset_.append(mem_split)
            nonmember_index_subset_.append(nonmem_split)

            # Create subset dataclass
            member_dset = ch.utils.data.Subset(train_data, mem_split)
            nonmember_dset = ch.utils.data.Subset(train_data, nonmem_split)
            # and loaders for subsequent use
            member_loader_ = ch.utils.data.DataLoader(
                member_dset, batch_size=batch_size, shuffle=False
            )
            nonmember_loader_ = ch.utils.data.DataLoader(
                nonmember_dset, batch_size=batch_size, shuffle=False
            )
            member_loader.append(member_loader_)
            nonmember_loader.append(nonmember_loader_)
        
        # Replace back train_index_subset and nonmember_index_subset
        train_index_subset = train_index_subset_
        nonmember_index_subset = nonmember_index_subset_
    else:
        member_dset = ch.utils.data.Subset(train_data, train_index_subset)
        nonmember_dset = ch.utils.data.Subset(
            train_data,
            nonmember_index_subset,
        )

        # Make loaders out of data
        member_loader = ch.utils.data.DataLoader(
            member_dset, batch_size=batch_size, shuffle=False
        )
        nonmember_loader = ch.utils.data.DataLoader(
            nonmember_dset, batch_size=batch_size, shuffle=False
        )

    if want_all_member_nonmember:
        return member_loader, nonmember_loader
    return (
        member_loader,
        nonmember_loader,
        nonmember_dset_ft,
        train_index_subset,
        nonmember_index_subset,
    )


def load_ref_models(model_dir, args, num_classes: int):
    if args.same_seed_ref:
        folder_to_look_in = os.path.join(model_dir, f"same_init/{args.target_model_index}")
    else:
        folder_to_look_in = model_dir

    if args.specific_ref_folder is not None:
        folder_to_look_in = os.path.join(folder_to_look_in, args.specific_ref_folder)

    # Look specifically inside folder corresponding to this model's seed
    ref_models, ref_indices = [], []
    for m in os.listdir(folder_to_look_in):
        # Skip if directory
        if os.path.isdir(os.path.join(folder_to_look_in, m)):
            continue

        # Skip ref model if trained on exact same data split as target model
        # if m.split(".pt")[0].split("_")[0] == f"{args.target_model_index}":
        #    continue

        model, _, _ = get_model(args.model_arch, num_classes)
        state_dict = ch.load(os.path.join(folder_to_look_in, m))
        ref_indices.append(state_dict["train_index"])
        model.load_state_dict(state_dict["model"], strict=False)
        model.eval()
        ref_models.append(model)

    ref_indices = np.array(ref_indices, dtype=object)
    return ref_models, ref_indices


def get_signals(return_dict,
                job_index: int,
                args,
                attacker, loader, ds,
                is_train: bool,
                nonmember_dset_ft,
                model_dir: str,
                model_map: np.ndarray,
                learning_rate: float,
                num_samples: int,
                ref_models = None,):
    # Weird multiprocessing bug when using a dill wrapper, so need to re-import
    from tqdm import tqdm
    import numpy as np

    # Compute signals for member data
    signals = []
    for i, (x, y) in tqdm(enumerate(loader),
                          total=len(loader),
                          position = job_index,
                          desc="Member" if is_train else "Non-member"):
        out_models_use = None
        in_models_use = None
        if attacker.reference_based:
            in_models_use = [ref_models[j] for j in np.nonzero(model_map[:, i])[0]]
            if args.num_ref_models is not None:
                in_models_use = in_models_use[: args.num_ref_models]

            # For L-mode, load out models specific to datapoint
            if args.l_mode and is_train:
                this_dir = os.path.join(model_dir, f"l_mode/{i}")
                out_models_use, ref_indices = load_ref_models(this_dir, args)
            else:
                # Use existing ref models
                out_models_use = [
                    ref_models[j] for j in np.nonzero(1 - model_map[:, i])[0][:]
                ]
                if args.num_ref_models is not None:
                    out_models_use = out_models_use[: args.num_ref_models]
                    in_models_use = in_models_use[: args.num_ref_models]

        # Apply input augmentations
        x_aug = None
        if args.aug:
            x_aug = ds.get_augmented_input(x, y)

        score = attacker.compute_scores(
            x,
            y,
            out_models=out_models_use,
            in_models=in_models_use,
            other_data_source=nonmember_dset_ft,
            x_aug=x_aug,
            learning_rate=learning_rate,
            num_samples=num_samples,
            is_train=is_train,
            momentum=args.momentum
        )
        signals.append(score)
    prefix = "member" if is_train else "nonmember"
    return_dict[f"{prefix}_{job_index}"] = signals


def main(args):
    model_dir = os.path.join(get_models_path(),
                             args.dataset,
                             args.model_arch,
                             f"lr_{args.momentum}_wd_{args.weight_decay}")

    ds = get_dataset(args.dataset)(augment=False)

    # Load target model
    target_model, criterion, hparams = get_model(args.model_arch, ds.num_classes)
    model_dict = ch.load(os.path.join(model_dir, f"{args.target_model_index}.pt"), map_location="cpu")
    target_model.load_state_dict(model_dict["model"], strict=False)
    target_model.eval()

    # Pick records (out of all train) to test
    train_index = model_dict["train_index"]

    # Get some information about model
    learning_rate = hparams['learning_rate']
    num_samples = len(train_index)

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
            for (x, y) in tqdm(loader, desc=f"Collecting features from layer {args.partial_model}"):
                features = target_model(x.cuda(), layer_readout=args.partial_model).cpu().detach()
                X_.append(features)
                Y_.append(y)
        # Make a new dataset out of X_ and Y_
        target_model.cpu()
        train_data = ch.utils.data.TensorDataset(ch.cat(X_), ch.cat(Y_))

    # Get data split
    (
        member_loader,
        nonmember_loader,
        nonmember_dset_ft,
        train_index_subset,
        nonmember_index_subset,
    ) = member_nonmember_loaders(
        train_data,
        train_index,
        args.num_points,
        args.exp_seed,
        num_nontrain_pool=num_nontrain_pool,
        split_each_loader=args.num_parallel_each_loader,
    )

    hessian = None
    # If attack uses uses_hessian, try loading from disk if available
    hessian_store_path = os.path.join(
        get_misc_path(),
        args.dataset,
        f"{args.model_arch}_lr_{args.momentum}_wd_{args.weight_decay}",
        str(args.target_model_index),
    )
    # Load Hessian if exists (unless approximate iHVP is used)
    if os.path.exists(os.path.join(hessian_store_path, "hessian.ch")) and not args.approximate_ihvp:
        hessian = ch.load(os.path.join(hessian_store_path, "hessian.ch"))
        print("Loaded Hessian!")

    # Repeat each entry in num_gpus by (num_parallel_each_loader * 2) times
    num_gpus = ch.cuda.device_count()
    gpu_assignments = np.tile(np.arange(num_gpus), args.num_parallel_each_loader * 2)

    train_index_for_hessian_loader = train_index
    if args.subsample_ihvp < 1:
        np.random.seed(args.exp_seed + 5)
        train_index_for_hessian_loader = np.random.choice(
            train_index, int(len(train_index) * args.subsample_ihvp), replace=False
        )

    # Extract "later" part of model
    if args.partial_model is not None:
        model_for_ihvp = target_model.make_later_layer_model(args.partial_model + 1)
        model_for_ihvp.eval()
    else:
        model_for_ihvp = target_model

    # For reference-based attacks, train out models
    attackers_mem, attackers_nonmem = [], []
    for i in range(args.num_parallel_each_loader):
        # Keep cycling across num-GPUs and assign each odd to mem...
        attacker_mem = get_attack(args.attack)(
            copy.deepcopy(model_for_ihvp),
            criterion,
            all_train_loader=get_loader(
                train_data,
                train_index_for_hessian_loader,
                args.hessian_batch_size,
                num_workers=0),
            hessian=hessian,
            damping_eps=args.damping_eps,
            low_rank=args.low_rank,
            save_compute_trick=args.save_compute_trick,
            approximate=args.approximate_ihvp,
            tol=args.cg_tol,
            weight_decay=args.weight_decay,
            skip_reg_term=args.skip_reg_term,
            device=f"cuda:{gpu_assignments[i]}",
        )
        attackers_mem.append(attacker_mem)

        # ...and each even to non-mem
        attacker_nonmem = get_attack(args.attack)(
            copy.deepcopy(model_for_ihvp),
            criterion,
            all_train_loader=get_loader(
                train_data,
                train_index_for_hessian_loader,
                args.hessian_batch_size,
                num_workers=0,
            ),
            hessian=hessian,
            damping_eps=args.damping_eps,
            low_rank=args.low_rank,
            approximate=args.approximate_ihvp,
            save_compute_trick=args.save_compute_trick,
            tol=args.cg_tol,
            weight_decay=args.weight_decay,
            skip_reg_term=args.skip_reg_term,
            device=f"cuda:{gpu_assignments[i + args.num_parallel_each_loader]}",
        )
        attackers_nonmem.append(attacker_nonmem)

    # Shared dict to get reutnr values
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []

    single_process_mode = False
    if args.num_parallel_each_loader == 1:
        single_process_mode = True
        member_loader = [member_loader]
        nonmember_loader = [nonmember_loader]
        train_index_subset = [train_index_subset]
        nonmember_index_subset = [nonmember_index_subset]

    # Load up reference models
    if attackers_mem[0].reference_based and not args.l_mode:
        ref_models, ref_indices = load_ref_models(model_dir, args, ds.num_classes)
    else:
        member_map = None
        nonmember_map = None
        ref_models = None

    for i in range(args.num_parallel_each_loader):

        # Select correct reference models map
        if ref_models is not None:
            # For each reference model, look at ref_indices and create a 'isin' based 2D bool-map
            # Using train_index_subset
            member_map    = np.zeros((len(ref_models), len(train_index_subset[i])),     dtype=bool)
            nonmember_map = np.zeros((len(ref_models), len(nonmember_index_subset[i])), dtype=bool)
            for j, out_index in enumerate(ref_indices):
                member_map[j]    = np.isin(train_index_subset[i], out_index)
                nonmember_map[j] = np.isin(nonmember_index_subset[i], out_index)

        # Process for member data
        p = DillProcess(
            target=get_signals,
            args=(
                return_dict,
                i,
                args,
                attackers_mem[i],
                member_loader[i],
                ds,
                True,
                nonmember_dset_ft,
                model_dir,
                member_map,
                learning_rate,
                num_samples,
                ref_models,
            ),
        )
        p.start()
        processes.append(p)
        # Process for non-member data
        p = DillProcess(
            target=get_signals,
            args=(
                return_dict,
                i + args.num_parallel_each_loader,
                args,
                attackers_nonmem[i],
                nonmember_loader[i],
                ds,
                False,
                nonmember_dset_ft,
                model_dir,
                nonmember_map,
                learning_rate,
                num_samples,
                ref_models,
            ),
        )
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Extract relevant data
    # Everything starting with "member_" is for member data, and "nonmember_" for non-member data
    signals_in, signals_out = [], []
    for k, v in return_dict.items():
        if k.startswith("member_"):
            signals_in.extend(v)
        else:
            signals_out.extend(v)

    """
    # Compute signals for member data
    signals_in, signals_out = [], []
    for i, (x, y) in tqdm(enumerate(member_loader), total=len(member_loader)):
        out_models_use = None
        in_models_use = None
        out_traces_use = None
        if attacker.reference_based:
            in_models_use = [ref_models[j] for j in np.nonzero(member_map[:, i])[0]]
            if args.num_ref_models is not None:
                in_models_use = in_models_use[: args.num_ref_models]

            # For L-mode, load out models specific to datapoint
            if args.l_mode:
                this_dir = os.path.join(model_dir, f"l_mode/{i}")
                out_models_use, ref_indices = load_ref_models(this_dir, args)
            else:
                # Use existing ref models
                out_models_use = [ref_models[j] for j in np.nonzero(1 - member_map[:, i])[0][:]]
                if attacker.requires_trace:
                    out_traces_use = [ref_traces[j] for j in np.nonzero(1 - member_map[:, i])[0]]
                if args.num_ref_models is not None:
                    out_models_use = out_models_use[: args.num_ref_models]
                    in_models_use = in_models_use[: args.num_ref_models]
                    if attacker.requires_trace:
                        out_traces_use = out_traces_use[: args.num_ref_models]

        # Apply input augmentations
        x_aug = None
        if args.aug:
            x_aug = ds.get_augmented_input(x, y)

        score = attacker.compute_scores(
            x,
            y,
            out_models=out_models_use,
            in_models=in_models_use,
            other_data_source=nonmember_dset_ft,
            out_traces=out_traces_use,
            x_aug=x_aug,
            learning_rate=learning_rate,
            num_samples=num_samples,
            is_train=True
        )
        signals_in.append(score)

    # Compute signals for non-member data
    for i, (x, y) in tqdm(enumerate(nonmember_loader), total=len(nonmember_loader)):
        out_models_use = None
        in_models_use = None
        if attacker.reference_based:
            # TODO: train out models for L-mode for non-members
            out_models_use = [ref_models[j] for j in np.nonzero(1 - nonmember_map[:, i])[0]]
            in_models_use  = [ref_models[j] for j in np.nonzero(nonmember_map[:, i])[0]]
            if args.num_ref_models is not None:
                out_models_use = out_models_use[: args.num_ref_models]
                in_models_use = in_models_use[: args.num_ref_models]

        # Apply input augmentations
        x_aug = None
        if args.aug:
            x_aug = ds.get_augmented_input(x, y)

        score = attacker.compute_scores(
            x,
            y,
            out_models=out_models_use,
            in_models=in_models_use,
            other_data_source=nonmember_dset_ft,
            x_aug=x_aug,
            learning_rate=learning_rate,
            num_samples=num_samples,
            is_train=False
        )
        signals_out.append(score)
    """

    # Save signals
    signals_in  = np.array(signals_in).flatten()
    signals_out = np.array(signals_out).flatten()
    signals_dir = get_signals_path()
    save_dir = os.path.join(
        signals_dir,
        args.dataset,
        f"{args.model_arch}_lr_{args.momentum}_wd_{args.weight_decay}",
        str(args.target_model_index),
    )

    # Make sure save_dir exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    attack_name = args.attack
    suffix = ""
    if args.same_seed_ref:
        attack_name += "_same_seed_ref"
    if args.num_ref_models is not None:
        attack_name += f"_{args.num_ref_models}_ref"
    if args.aug:
        attack_name += "_aug"
    if attackers_mem[0].uses_hessian:
        attack_name += f"_damping_{args.damping_eps}_lowrank_{args.low_rank}"
    if args.approximate_ihvp:
        attack_name += f"_approx_ihvp_{args.damping_eps}"

    if args.sif_proper_mode:
        attack_name += "_sif_proper_mode"
    if args.suffix is not None:
        suffix = f"_{args.suffix}"
    if args.simulate_metaclf:
        attack_name += "_metaclf"

    if args.simulate_metaclf:
        # Use a 2-depth decision tree to fit a meta-classifier
        signals_in_, signals_out_ = [], []
        for i in tqdm(range(len(signals_in)), desc="Meta-Clf(In)"):
            # Concatenate all data except signals_in[i], along with signals_out
            # With appropriate labels
            X = np.concatenate((signals_in[:i], signals_in[i + 1 :], signals_out)).reshape(-1, 1)
            Y = np.concatenate((np.ones(len(signals_in) - 1), np.zeros(len(signals_out))))
            # Use randomforest
            clf = RandomForestClassifier(max_depth=2)
            clf.fit(X, Y)
            # Get prediction for signals_in[i]
            pred = clf.predict_proba(signals_in[i].reshape(1, -1))[0][1]
            signals_in_.append(pred)
        # Repeat same for signals_out
        for i in tqdm(range(len(signals_out)), desc="Meta-Clf(Out)"):
            X = np.concatenate((signals_out[:i], signals_out[i + 1 :], signals_in)).reshape(-1, 1)
            Y = np.concatenate((np.zeros(len(signals_out) - 1), np.ones(len(signals_in))))
            clf = RandomForestClassifier(max_depth=2)
            clf.fit(X, Y)
            pred = clf.predict_proba(signals_out[i].reshape(1, -1))[0][1]
            signals_out_.append(pred)
        # Replace with these scores
        signals_in = np.array(signals_in_)
        signals_out = np.array(signals_out_)

    # Print out ROC
    total_labels = [0] * len(signals_out) + [1] * len(signals_in)
    total_preds = np.concatenate((signals_out, signals_in))

    # If SIF attack, need to perform thresholding
    # Consider extreme case of actual work, where we consider n-1 setting
    if args.attack == "SIF" and args.sif_proper_mode:
        total_preds_ = []

        labels_ = np.array([0] * (len(signals_out) - 1) + [1] * len(signals_in))
        for i, (x, y) in tqdm(enumerate(nonmember_loader), total=len(nonmember_loader)):
            logit = target_model(x.cuda())
            if type(criterion).__name__ == "MSELoss":
                pred = (logit.squeeze(1) > 0.5).float().cpu()
            else:
                pred = logit.argmax(dim=1).cpu()
            if pred != y:
                total_preds_.append(0.0)
            else:
                scores_ = np.concatenate(
                    (signals_out[:i], signals_out[i + 1 :], signals_in)
                )
                min_t, max_t = attacker_nonmem.get_thresholds(scores_, labels_)
                if min_t < signals_out[i] and signals_out[i] < max_t:
                    total_preds_.append(1.0)
                else:
                    total_preds_.append(0.0)

        # Repeat for member target data
        labels_ = np.array([0] * len(signals_out) + [1] * (len(signals_in) - 1))
        for i, (x, y) in tqdm(enumerate(member_loader), total=len(member_loader)):
            logit = target_model(x.cuda())
            # If MSE loss
            if type(criterion).__name__ == "MSELoss":
                pred = (logit.squeeze(1) > 0.5).float().cpu()
            else:
                pred = logit.argmax(dim=1).cpu()
            if pred != y:
                total_preds_.append(0.)
            else:
                scores_ = np.concatenate((signals_out, signals_in[:i], signals_in[i + 1 :]))
                min_t, max_t = attackers_mem[0].get_thresholds(scores_, labels_)
                if min_t < signals_in[i] and signals_in[i] < max_t:
                    total_preds_.append(1.)
                else:
                    total_preds_.append(0.)

        # Proceed to use these scores for MIA
        # Replace total_preds
        total_preds = np.array(total_preds_)
        # Because these scores are 0/1, TPR/FPR will be 0/1 as well
        # Choose to focus on AUC instead

    fpr, tpr, _ = roc_curve(total_labels, total_preds)
    roc_auc = auc(fpr, tpr)
    print("\n\n\n\nAUC: %.3f" % roc_auc)

    # Save results
    np.save(
        f"{save_dir}/{attack_name}{suffix}.npy",
        {
            "in": signals_in,
            "out": signals_out,
        },
    )

    # Save Hessian, if computed and didn't exist before
    if attackers_mem[0].uses_hessian and (
        not os.path.exists(os.path.join(hessian_store_path, "hessian.ch"))
    ):
        os.makedirs(hessian_store_path, exist_ok=True)
        ch.save(
            attackers_mem[0].get_hessian(), os.path.join(hessian_store_path, "hessian.ch")
        )
        print("Saved Hessian!")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_arch", type=str, default="wide_resnet_28_2")
    args.add_argument("--dataset", type=str, default="cifar10")
    args.add_argument("--attack", type=str, default="LOSS")
    args.add_argument("--hessian_batch_size", type=int, default=256, help="Batch size for Hessian computation")
    args.add_argument("--exp_seed", type=int, default=2024)
    args.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer.")
    args.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay for SGD optimizer.")
    args.add_argument("--damping_eps", type=float, default=2e-1, help="Damping for Hessian computation (only valid for some attacks)")
    args.add_argument("--subsample_ihvp", type=float, default=1.0, help="If < 1, use a fraction of actual train data to pass to Hessian")
    args.add_argument(
        "--cg_tol",
        type=float,
        default=1e-5,
        help="Tol for iHVP in CG (when using Approx)",
    )
    args.add_argument(
        "--approximate_ihvp",
        action="store_true",
        help="If true, use approximate iHV (using CG) instead of exact iHVP",
    )
    args.add_argument(
        "--skip_reg_term",
        action="store_true",
        help="If true, skip terms I4/I5 in Hessian computation for regularization-baesd attacks",
    )
    args.add_argument(
        "--save_compute_trick",
        action="store_true",
        help="If true, use trick to skip computing H-1\grad(L0) for each target record.",
    )
    args.add_argument(
        "--low_rank",
        action="store_true",
        help="If true, use low-rank approximation of Hessian. Else, use damping. Useful for inverse",
    )
    args.add_argument(
        "--partial_model",
        type=int,
        default=None,
        help="Part of model from this specified layer is treated as actual model. Helps in computation for Hessian-based attacks",
    )
    args.add_argument("--target_model_index", type=int, default=0, help="Index of target model")
    args.add_argument("--num_ref_models", type=int, default=None)
    args.add_argument(
        "--simulate_metaclf",
        action="store_true",
        help="If true, use scores as features and fit LOO-style meta-classifier for each target datapoint",
    )
    args.add_argument("--l_mode", action="store_true", help="L-mode (where out reference model is trained on all data except target record)")
    args.add_argument("--aug", action="store_true", help="Use augmented data?")
    args.add_argument("--sif_proper_mode", action="store_true", help="Tune 2 thresholds for SIF like original paper")
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
    args.add_argument(
        "--num_parallel_each_loader",
        type=int,
        default=2,
        help="Split each loader into this many parts. Use > 1 for more parallelism",
    )
    args = args.parse_args()

    mp.set_start_method('spawn')
    main(args)
