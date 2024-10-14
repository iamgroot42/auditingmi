import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import argparse
import seaborn as sns
import torch as ch
from collections import defaultdict

from mib.utils import get_signals_path

import matplotlib as mpl


def main(args):
    signals_path = os.path.join(
        get_signals_path(),
        args.dataset,
        f"{args.model_arch}_lr_{args.momentum}_wd_{args.weight_decay}",
        f"{args.target_model_index}",
    )

    predictions_at_given_fpr = {"member": {}, "non_member": {}}

    # Ground-truth
    predictions_at_given_fpr["member"]["GT"] = np.ones(args.num_points)
    predictions_at_given_fpr["non_member"]["GT"] = np.zeros(args.num_points)

    target_fpr = args.target_fpr

    # Load all other attacks
    for attack in os.listdir(signals_path):
        if not attack.endswith(f"_npoints_{args.num_points}_l_mode.npy"):
            continue
        attack_name = attack.split(f"_npoints_{args.num_points}_l_mode")[0]

        data = np.load(os.path.join(signals_path, attack), allow_pickle=True).item()

        signals_in = data["in"]
        signals_out = data["out"]
        total_labels = [0] * args.num_points + [1] * args.num_points

        total_preds = np.concatenate((signals_out, signals_in))
        fpr, tpr, thresholds = roc_curve(total_labels, total_preds)

        # Compute AUC as well, while at it
        auc_score = auc(fpr, tpr)
        print(f"{attack_name} AUC: {auc_score}")

        # Thresholds
        fpr_threshold = thresholds[np.where(np.array(fpr) < target_fpr)[0][-1]]
        # Predictions for members
        member_preds_at_given_fpr = signals_in >= fpr_threshold
        # Predictions for non-members
        non_member_preds_at_given_fpr = signals_out >= fpr_threshold
        # Store them
        predictions_at_given_fpr["member"][attack_name] = member_preds_at_given_fpr
        predictions_at_given_fpr["non_member"][attack_name] = non_member_preds_at_given_fpr

    print("\n\n######\n\n")
    # Load up file for L-attack predictions, directly read out predictions for desired FPR
    loo_dict = ch.load(f"./predictions_{args.dataset}_{args.model_arch}_{args.num_points}_loo.pth")
    actual_fprs = np.array(list(loo_dict.keys()))

    # Also print out TPR at 1% and 0.1% FPR
    def get_tpr(upper_bound):
        return actual_fprs[np.where(actual_fprs < upper_bound)[0][-1]]

    actual_fpr_given = get_tpr(target_fpr)
    predictions_at_given_fpr["non_member"]["L-attack"] = loo_dict[actual_fpr_given][:args.num_points]
    predictions_at_given_fpr["member"]["L-attack"] = loo_dict[actual_fpr_given][args.num_points:]

    # If Lira-L predictions available, load them too
    lira_l_path = f"./predictions_{args.dataset}_{args.model_arch}_{args.num_points}_loo_lira_offline_simulate.pth"
    if os.path.exists(lira_l_path):
        lira_loo_dict = ch.load(lira_l_path)
        actual_fprs = np.array(list(lira_loo_dict.keys()))

        actual_fpr_given = actual_fprs[np.where(actual_fprs < target_fpr)[0][-1]]

        predictions_at_given_fpr["non_member"]["LiRA-L"] = lira_loo_dict[actual_fpr_given][:args.num_points]
        predictions_at_given_fpr["member"]["LiRA-L"] = lira_loo_dict[actual_fpr_given][args.num_points:]

    # For target FPR, compute pair-wise agreement between each attack for members non-members
    attack_names = list(predictions_at_given_fpr["member"].keys())
    for i in range(len(attack_names)):
        for j in range(i+1, len(attack_names)):
            attack_1 = attack_names[i]
            attack_2 = attack_names[j]
            # Agreement for members at given FPR
            member_agreement_given_fpr = np.mean(predictions_at_given_fpr["member"][attack_1] == predictions_at_given_fpr["member"][attack_2])
            # Agreement for non-members at given FPR
            non_member_agreement_given_fpr = np.mean(predictions_at_given_fpr["non_member"][attack_1] == predictions_at_given_fpr["non_member"][attack_2])
            print(f"{attack_1} & {attack_2} | {100*target_fpr}%FPR: member {member_agreement_given_fpr}, non-member {non_member_agreement_given_fpr}")
        print("\n")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--target_model_index", type=int, default=0, help="Index of target model")
    args.add_argument("--model_arch", type=str, default="mlp2")
    args.add_argument("--num_points", type=int, default=500, help="Number of points in the signals.")
    args.add_argument("--dataset", type=str, default="purchase100_s")
    args.add_argument("--plotdir", type=str, default="./plots")
    args.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer.")
    args.add_argument("--target_fpr", type=float, help="FPR for which prediction similarity is to be studied")
    args.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay for SGD optimizer.",
    )
    args = args.parse_args()
    main(args)
