#!/bin/bash

DATASET=$1
MODEL=$2

for i in {1..9}
do
    # Run the attack
    python mib/attack.py --target_model_index $i --dataset $DATASET --model_arch $MODEL --weight_decay 0 --attack IHA --approximate_ihvp --num_points -1 --num_parallel_each_loader 16 --cg_tol 1e-3
done