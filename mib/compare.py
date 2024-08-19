import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import argparse
import seaborn as sns
from collections import defaultdict

from mib.utils import get_signals_path

import matplotlib as mpl


# Change DPI
mpl.rcParams["figure.dpi"] = 400

ATTACKS_TO_PLOT = [
    "LOSS",
    "LiRAOnline",
    # "SIF_damping_0.2_lowrank_False",
    "IHA_damping_0.2_lowrank_False",
    # "IHA_approx_ihvp_0.2_tol_0.001",
]
ATTACK_MAPPING = {
    "LOSS": "LOSS",
    "IHA": "IHA (Ours)",
    "IHA_damping_0.2_lowrank_False": "IHA (Ours)",
    "LiRAOnline": "LiRA",
    "SIF_damping_0.2_lowrank_False": "SIF",
    "IHA_approx_ihvp_0.2_tol_0.001": "IHA (CG)",
}
COLOR_MAPPING = {
    "IHA (Ours)": 0,
    "IHA (CG)": 1,
    "LiRA": 2,
    "LOSS": 3,
    "SIF": 4,
    "RMIA": 5,
}


def main(args):
    # signals_path = os.path.join(get_signals_path(), "unhinged_audit", str(args.model_index))
    signals_path = os.path.join(
        get_signals_path(),
        args.dataset,
        f"{args.model_arch}_lr_{args.momentum}_wd_{args.weight_decay}",
    )

    for model_index in os.listdir(signals_path):
        inside_model_index = os.path.join(signals_path, model_index)
        all_raw_preds_at_1p_fpr = defaultdict()
        all_raw_scores_at_1p_fpr = defaultdict()

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
            data = np.load(
                os.path.join(inside_model_index, attack), allow_pickle=True
            ).item()

            signals_in = data["in"]
            signals_out = data["out"]

            if abs(len(signals_in) - len(signals_out)) > 1:
                raise ValueError(
                    f"Mismatched signals for {attack_name}[{model_index}]: found {len(signals_in)} signals_in and {len(signals_out)} signals_out."
                )

            # Accidentally forgot negative sign for FLIP signals. Will fix later, but for now just flip the signals here
            if "SIF" in attack_name:
                signals_in = -signals_in
                signals_out = -signals_out

            total_labels = [0] * len(signals_out) + [1] * len(signals_in)

            total_preds = np.concatenate((signals_out, signals_in))
            fpr, tpr, thresholds = roc_curve(total_labels, total_preds)

            # Threshold that gives 1% FPR
            one_p_fpr_threshold = thresholds[np.where(np.array(fpr) < 0.01)[0][-1]]
            # Predictions for members at this threshold
            member_preds_at_1p_fpr = signals_in >= one_p_fpr_threshold
            all_raw_preds_at_1p_fpr[attack_name] = member_preds_at_1p_fpr
            all_raw_scores_at_1p_fpr[attack_name] = signals_in

        preds_as_array = np.array(list(all_raw_preds_at_1p_fpr.values()))
        scores_as_array = np.array(list(all_raw_scores_at_1p_fpr.values()))

        # Look at cases where IHA is righta
        iha_right = preds_as_array[-1] == 1
        # Look at LOSS values for iha_right and ~iha_right
        iha_loss_when_correct = -scores_as_array[0][iha_right]
        iha_loss_when_incorrect = -scores_as_array[0][~iha_right]

        lira_correct = preds_as_array[1] == 1
        lira_loss_when_correct = -scores_as_array[0][lira_correct]
        lira_loss_when_incorrect = -scores_as_array[0][~lira_correct]

        # Plot these 2 distributions together
        plt.hist(iha_loss_when_correct, bins=100, alpha=0.5, label="IHA correct")
        plt.hist(iha_loss_when_incorrect, bins=100, alpha=0.5, label="IHA incorrect")
        # plt.hist(lira_loss_when_correct, bins=100, alpha=0.5, label="LiRA correct")
        # plt.hist(lira_loss_when_incorrect, bins=100, alpha=0.5, label="LiRA incorrect")
        # plt.xscale("log")
        plt.legend()
        plt.savefig("loss_distributions.png")
        exit(0)

        # Filter out records where all attacks agree on labels (as 1 or 0)
        agreement = np.all(preds_as_array, axis=0) | np.all(1 - preds_as_array, axis=0)
        preds_as_array = preds_as_array[:, ~agreement]
        # Now plot a binary heatmap of the predictions in seaborn
        # Provide name per row (attack) as well
        sns.heatmap(
            preds_as_array,
            yticklabels=[ATTACK_MAPPING[attack] for attack in all_raw_preds_at_1p_fpr.keys()],
            cmap="coolwarm",
        )
        plt.savefig("prediction_heatmap.png")
        exit(0)

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
                attack_name,
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
        plt.plot(
            fprs,
            tprs,
            label=ATTACK_MAPPING[attack_name],
            c=CB_colors[COLOR_MAPPING[ATTACK_MAPPING[attack_name]]],
        )
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
            custom_lines.append(
                plt.Line2D([0], [0], color=CB_colors[COLOR_MAPPING[k]], lw=2)
            )
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


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_arch", type=str, default="wide_resnet_28_2")
    args.add_argument("--dataset", type=str, default="cifar10")
    args.add_argument("--plotdir", type=str, default="./plots")
    args.add_argument(
        "--which_plot",
        type=int,
        default=0,
        help="Plot TPR-FPR curves for this specific model",
    )
    args.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum for SGD optimizer."
    )
    args.add_argument("--noplot", action="store_true", help="Don't plot the ROC curve.")
    args.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay for SGD optimizer.",
    )
    args = args.parse_args()
    main(args)
