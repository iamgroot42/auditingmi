import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import argparse
from collections import defaultdict

from mib.utils import get_signals_path

import matplotlib as mpl


ATTACKS_TO_PLOT = [
    # "LiRAOnline",
    # "LiRAOffline",
    # "LiRAOnline_aug",
    "LOSS",
    # "GradNorm",
    "LiRAOnline",
    # "LiRAOnline_same_seed_ref",
    # "LiRAOnline_same_seed_ref_aug",
    # "LiRAOnline_20_ref_aug",
    # "LiRAOnline_same_seed_ref_20_ref",
    # "LiRAOnline_same_seed_ref_last5",
    # "LiRAOnline_same_seed_ref_aug_last5",
    # "UnlearnGradNorm",
    # "UnlearningAct",
    # "LiRAOnline_best5",
    # "LiRAOnline_aug_best5",
    # "LiRAOnline_last5",
    # "LiRAOnline_aug_last5",
    # "Activations",
    # "ActivationsOffline",
    # "IHA",
    "SIF_damping_0.2_lowrank_False",
    "Reference",
    "IHA_damping_0.2_lowrank_False",
    "IHA_damping_0.1_lowrank_False",
    "IHA_damping_0.01_lowrank_False",
    "IHA_approx_ihvp_0.2_tol_0.001",
    "IHA_approx_ihvp_0.2_tol_0.0001",
    "IHA_approx_ihvp_0.2_tol_1e-05",
    "IHA_approx_ihvp_0.2_tol_0.001_subsample_0.2",
    "IHA_approx_ihvp_0.2_tol_0.001_subsample_0.4",
    "IHA_approx_ihvp_0.2_tol_0.001_subsample_0.6",
    "IHA_approx_ihvp_0.2_tol_0.001_subsample_0.8",
    "IHA_approx_ihvp_0.2_tol_0.001_subsample_0.9",
    "IHA_damping_0.2_lowrank_False_npoints_5000",
    "IHA_approx_ihvp_1.0_tol_0.001_npoints_5000",
    "IHA_approx_ihvp_0.5_tol_0.001_npoints_5000",
    "IHA_damping_0.2_lowrank_False_skip_reg_term",
    "IHA_damping_0.2_lowrank_False_skip_loss",
    "IHA_damping_0.2_lowrank_False_skip_reg_term_skip_loss",
    "IHA_damping_0.2_lowrank_False_only_i1",
    "IHA_damping_0.2_lowrank_False_only_i2",
    "Reference_npoints_500_l_mode",
    "IHA_damping_0.2_lowrank_False_only_i1_include_loss",
    "IHA_damping_0.2_lowrank_False_only_i2_include_loss",
]
ATTACK_MAPPING = {
    "LOSS": "LOSS",
    "Reference": "Reference",
    "IHA": "IHA (Ours)",
    "IHA_damping_0.2_lowrank_False": "IHA (Ours)",
    "IHA_damping_0.1_lowrank_False": "IHA (0.1)",
    "IHA_damping_0.01_lowrank_False": "IHA (0.01)",
    "LiRAOnline": "LiRA",
    "SIF_damping_0.2_lowrank_False": "SIF",
    "IHA_approx_ihvp_0.2_tol_0.001": "IHA (CG)",
    "IHA_approx_ihvp_0.2_tol_0.0001": "IHA (CG, 1e-4)",
    "IHA_approx_ihvp_0.2_tol_1e-05": "IHA (CG, 1e-5)",
    "IHA_approx_ihvp_0.2_tol_0.001_subsample_0.2": "IHA (CG, 0.2)",
    "IHA_approx_ihvp_0.2_tol_0.001_subsample_0.4": "IHA (CG, 0.4)",
    "IHA_approx_ihvp_0.2_tol_0.001_subsample_0.6": "IHA (CG, 0.6)",
    "IHA_approx_ihvp_0.2_tol_0.001_subsample_0.8": "IHA (CG, 0.8)",
    "IHA_approx_ihvp_0.2_tol_0.001_subsample_0.9": "IHA (CG, 0.9)",
    "IHA_damping_0.2_lowrank_False_npoints_5000": "IHA (Ours, 5000)",
    "IHA_approx_ihvp_1.0_tol_0.001_npoints_5000": "IHA (Ours, 5000, 1e0)",
    "IHA_approx_ihvp_0.5_tol_0.001_npoints_5000": "IHA (Ours, 5000, 5e-1)",
    "IHA_damping_0.2_lowrank_False_skip_reg_term": "IHA (no I3,I4)",
    "IHA_damping_0.2_lowrank_False_skip_loss": "IHA (no loss)",
    "IHA_damping_0.2_lowrank_False_skip_reg_term_skip_loss": "IHA (no I3,I4, loss)",
    "IHA_damping_0.2_lowrank_False_only_i1": "IHA (only I1)",
    "IHA_damping_0.2_lowrank_False_only_i2": "IHA (only I2)",
    "Reference_npoints_500_l_mode": "LOO (500 points)",
    "IHA_damping_0.2_lowrank_False_only_i1_include_loss": "IHA (I1 and loss)",
    "IHA_damping_0.2_lowrank_False_only_i2_include_loss": "IHA (I2 and loss)",
}
COLOR_MAPPING = {
    "IHA (Ours)": 0,
    "IHA (CG)": 1,
    "LiRA": 2,
    "LOSS": 3,
    "SIF": 4,
    "RMIA": 5
}


def main(args):
    # signals_path = os.path.join(get_signals_path(), "unhinged_audit", str(args.model_index))
    signals_path = os.path.join(get_signals_path(), args.dataset, f"{args.model_arch}_lr_{args.momentum}_wd_{args.weight_decay}")

    info = defaultdict(list)
    for model_index in os.listdir(signals_path):
        inside_model_index = os.path.join(signals_path, model_index)

        for attack in os.listdir(inside_model_index):
            # Remove ".ch" from end
            attack_name = attack[:-4]
            """
            if int(model_index) in [3, 4, 5, 6, 8] and "IHA" in attack_name:
                print("SKIPPED ONE!")
                continue
            """
            if attack_name not in ATTACKS_TO_PLOT:
                # print("Skipping", attack_name, "...")
                continue
            # """
            data = np.load(os.path.join(inside_model_index, attack), allow_pickle=True).item()

            signals_in = data["in"]
            signals_out = data["out"]

            if abs(len(signals_in) - len(signals_out)) > 1:
                raise ValueError(f"Mismatched signals for {attack_name}[{model_index}]: found {len(signals_in)} signals_in and {len(signals_out)} signals_out.")

            # Accidentally forgot negative sign for FLIP signals. Will fix later, but for now just flip the signals here
            if "SIF" in attack_name:
                signals_in = -signals_in
                signals_out = -signals_out

            total_labels = [0] * len(signals_out) + [1] * len(signals_in)

            total_preds = np.concatenate((signals_out, signals_in))

            fpr, tpr, thresholds = roc_curve(total_labels, total_preds)
            roc_auc = auc(fpr, tpr)

            tpr_at_low_fpr = lambda x: tpr[np.where(np.array(fpr) < x)[0][-1]]
            info_ = {
                "roc_auc": roc_auc,
                # "tpr@0.1fpr": tpr_at_low_fpr(0.1),
                "tpr%%@0.01fpr": tpr_at_low_fpr(0.01) * 100,
                "tpr%%@0.001fpr": tpr_at_low_fpr(0.001) * 100,
                "tpr%%@0.0001fpr": tpr_at_low_fpr(0.0001) * 100,
                "fpr": fpr,
                "tpr": tpr,
            }
            """
            if model_index == "0":
                print(ATTACK_MAPPING[attack_name], info_["roc_auc"], info_["tpr%%@0.01fpr"], info_["tpr%%@0.001fpr"])
            """
            info[attack_name].append(info_)
    # exit(0)

    # Print out attacks that have < 10 models for results
    for attack_name, info_ in info.items():
        # if len(info_) < 10:
        if len(info_) < 3 and "LOO" not in ATTACK_MAPPING[attack_name]:
            print(f"Ignore {attack_name} since it has results only for {len(info_)} models")
    print("\n\n")

    # Set colorblind-friendly colors
    CB_colors = [
        "#377eb8",
        "#ff7f00",
        "#4daf4a",
        "#f781bf",
        "#a65628",
        "#984ea3",
        "#999999",
        "#e41a1c",
        "#dede00",
    ]

    # Font-type and figure DPI
    mpl.rcParams["figure.dpi"] = 500
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = "Times New Roman"

    # Increase font
    plt.rcParams.update({"font.size": 12})
    # Increase font of axes and their labels
    plt.rcParams.update({"axes.labelsize": 14})
    plt.rcParams.update({"xtick.labelsize": 14})
    plt.rcParams.update({"ytick.labelsize": 14})

    # Aggregate results across models
    actually_plotted = []
    for attack_name, info_ in info.items():
        mean_auc = np.mean([i["roc_auc"] for i in info_])
        print(
            # "%s | AUC = %0.3f ±  %0.3f | TPR@0.1FPR=%0.3f ± %0.3f | TPR@0.01FPR=%0.3f ± %0.3f | TPR@0.001FPR=%0.3f ± %0.3f"
            "%s | AUC = %0.3f ±  %0.3f | TPR%%@1%%FPR=%0.2f ± %0.2f | TPR%%@0.1%%FPR=%0.2f ± %0.2f | TPR%%@0.01%%FPR=%0.2f ± %0.2f"
            % (
                ATTACK_MAPPING[attack_name],
                mean_auc,
                np.std([i["roc_auc"] for i in info_]),
                # np.mean([i["tpr@0.1fpr"] for i in info_]),
                # np.std([i["tpr@0.1fpr"] for i in info_]),
                np.mean([i["tpr%%@0.01fpr"] for i in info_]),
                np.std([i["tpr%%@0.01fpr"] for i in info_]),
                np.mean([i["tpr%%@0.001fpr"] for i in info_]),
                np.std([i["tpr%%@0.001fpr"] for i in info_]),
                np.mean([i["tpr%%@0.0001fpr"] for i in info_]),
                np.std([i["tpr%%@0.0001fpr"] for i in info_]),
            )
        )
        if ATTACK_MAPPING[attack_name] not in COLOR_MAPPING:
            print(f"\tWARNING: {attack_name} not in COLOR_MAPPING. Skipping")
            continue

        if args.noplot:
            continue

        fprs = info_[args.which_plot]["fpr"]
        tprs = info_[args.which_plot]["tpr"]
        plt.plot(fprs, tprs, label=ATTACK_MAPPING[attack_name], c=CB_colors[COLOR_MAPPING[ATTACK_MAPPING[attack_name]]])
        actually_plotted.append(ATTACK_MAPPING[attack_name])

    if args.noplot:
        exit(0)

    # Make sure plot directory exists
    if not os.path.exists(os.path.join(args.plotdir, args.dataset)):
        os.makedirs(os.path.join(args.plotdir, args.dataset))

    plt.legend(loc="lower right")
    # Custom legend, list actually_plotted in order of COLOR_MAPPING
    custom_lines = []
    ordered_names = []
    for k in COLOR_MAPPING.keys():
        if k in actually_plotted:
            custom_lines.append(plt.Line2D([0], [0], color=CB_colors[COLOR_MAPPING[k]], lw=2))
            ordered_names.append(k)
    plt.legend(custom_lines, ordered_names, loc="lower right")

    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("TPR")
    plt.xlabel("FPR")
    plt.savefig(
        os.path.join(
            args.plotdir,
            args.dataset,
            f"{args.model_arch}_lr_{args.momentum}_wd_{args.weight_decay}_roc.pdf",
        )
    )

    # Also save low TPR/FPR region
    plt.xlim([1e-5, 1e0])
    plt.ylim([1e-5, 1e0])
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(
        os.path.join(
            args.plotdir,
            args.dataset,
            f"{args.model_arch}_lr_{args.momentum}_wd_{args.weight_decay}_roc_lowfpr.pdf",
        )
    )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_arch", type=str, default="wide_resnet_28_2")
    args.add_argument("--dataset", type=str, default="cifar10")
    args.add_argument("--plotdir", type=str, default="./plots")
    args.add_argument("--which_plot", type=int, default=0, help="Plot TPR-FPR curves for this specific model")
    args.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer.")
    args.add_argument("--noplot", action="store_true", help="Don't plot the ROC curve.")
    args.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay for SGD optimizer.")
    args = args.parse_args()
    main(args)
