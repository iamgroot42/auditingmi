#!/bin/bash

DATASET=$1
MODEL=$2
WEIGHTDECAY=$3
ATTACK=$4

for i in {0..9}
do
    # Run the attack
    python mib/attack.py --dataset $DATASET --model_arch $MODEL --num_points -1 --attack $ATTACK --target_model_index $i --num_parallel_each_loader 2 --weight_decay $WEIGHTDECAY
done