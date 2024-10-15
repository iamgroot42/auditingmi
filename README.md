# auditing_mi

This repository contains the code for the paper "Do Parameters Reveal More than Loss for Membership Inference?"

## Instructions

First install python dependencies
```
pip install -r requirements.txt
```

Then, install the package

```
pip install -e .
```

## Setting environment variables

You can either provide the following environment variables, or pass them via your config/CLI:

```
MIB_DATA_SOURCE: Path to data directory
MIB_CACHE_SOURCE: Path to save models, signals, and paths.
```

## Running the experiments

1. Train models. For more details on how to control dataset, model architecture, number of models, etc., please refer to the help message of the script.

```
python mib/train.py --dataset purchase100_s --model_arch mlp2 
```

To train for the leave-one-out setting, add `--l_mode_ref_model 0` to specify the model whose training data to use for the LOO setting, and which record to skip via `--l_mode_ref_model 0` (replace the `0` in both as needed). To train multiple models on the exact same data (serving as LOO model for `out` records, add `--l_out` flag).

2. Generate attack scores.

```
python mib/attack.py --dataset purchase100_s --model_arch mlp2 --num_points -1 --attack LiRAOnline
```

To get signals for only a subset of the train data, replace the `-1` above. To use our attack, replace `LiRAOnline` with `IHA`. Note that this will first generate and store the Hessian for that model, which can take a while. To use the approximate iHVP version, add the `--approximate_ihvp` flag.

3. Visualize results

```
python mib/visualize.py --dataset purchase100_s --model_arch mlp2
```

Print AUCs, TPR at certain FPRs, and generate ROC curves (for normal and low-FPR range). To skip plot generation, add `--noplot` flag.

## Additional experiments

To generate scores for the L-attack (LOO setting), use

```
python mib/attack_loo.py --dataset purchase100_s --model_arch mlp2
```

Then, to look at agreement rates between attacks (for e.g. for 5% FPR), use

```
python mib/agreement_rates.py --dataset purchase100_s --model_arch mlp2 --target_fpr 0.05
```
