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
    # "ProperTheoryRef",
    "SIF_damping_0.2_lowrank_False",
    "Reference",
    "ProperTheoryRef_damping_0.2_lowrank_False",
    "ProperTheoryRef_approx_ihvp",
]
ATTACK_MAPPING = {
    "LOSS": "LOSS",
    "Reference": "Reference",
    "ProperTheoryRef": "IHA (Ours)",
    "ProperTheoryRef_damping_0.2_lowrank_False": "IHA (Ours)",
    "LiRAOnline": "LiRA",
    "SIF_damping_0.2_lowrank_False": "SIF",
    "ProperTheoryRef_approx_ihvp": "IHA (Ours)",
}
COLOR_MAPPING = {
    "IHA (Ours)": 0,
    "LiRA": 1,
    "Reference": 2,
    "LOSS": 3,
    "SIF": 4
}


def main(args):
    # signals_path = os.path.join(get_signals_path(), "unhinged_audit", str(args.model_index))
    signals_path = os.path.join(get_signals_path(), args.dataset, args.model_arch)

    info = defaultdict(list)
    for model_index in os.listdir(signals_path):
        inside_model_index = os.path.join(signals_path, model_index, f"lr_{args.momentum}_wd_{args.weight_decay}")

        for attack in os.listdir(inside_model_index):
            # Remove ".ch" from end
            attack_name = attack[:-4]
            """
            if int(model_index) in [3, 4, 5, 6, 8] and "ProperTheoryRef" in attack_name:
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
                "tpr@0.1fpr": tpr_at_low_fpr(0.1),
                "tpr@0.01fpr": tpr_at_low_fpr(0.01),
                "tpr@0.001fpr": tpr_at_low_fpr(0.001),
                "fpr": fpr,
                "tpr": tpr,
            }
            info[attack_name].append(info_)

    """
    # First, assert we have same # of models per attack
    num_models = [len(info[attack_name]) for attack_name in info.keys()]
    if len(set(num_models)) != 1:
        raise ValueError("Different number of models per attack:", {k:len(v) for k,v in info.items()})
    print("Number of models per attack:", num_models[0])
    """

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
            "%s | AUC = %0.3f ±  %0.3f | TPR@0.1FPR=%0.3f ± %0.3f | TPR@0.01FPR=%0.3f ± %0.3f | TPR@0.001FPR=%0.3f ± %0.3f"
            % (
                attack_name,
                mean_auc,
                np.std([i["roc_auc"] for i in info_]),
                np.mean([i["tpr@0.1fpr"] for i in info_]),
                np.std([i["tpr@0.1fpr"] for i in info_]),
                np.mean([i["tpr@0.01fpr"] for i in info_]),
                np.std([i["tpr@0.01fpr"] for i in info_]),
                np.mean([i["tpr@0.001fpr"] for i in info_]),
                np.std([i["tpr@0.001fpr"] for i in info_]),
            )
        )
        fprs = info_[args.which_plot]["fpr"]
        tprs = info_[args.which_plot]["tpr"]
        plt.plot(fprs, tprs, label=ATTACK_MAPPING[attack_name], c=CB_colors[COLOR_MAPPING[ATTACK_MAPPING[attack_name]]])
        actually_plotted.append(ATTACK_MAPPING[attack_name])

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
    args.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay for SGD optimizer.")
    args = args.parse_args()
    main(args)
