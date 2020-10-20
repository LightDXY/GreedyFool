#!/bin/bash

Iter=10000
EPS=$2
PHASE=BLACK
now=$(date +"%Y%m%d_%H%M%S")
NAME=GreedyFool_Black_E_${EPS}-Iter_${Iter}_$now
CUDA_VISIBLE_DEVICES=$1 python -u nips_black_gd.py \
    --dataroot /mnt/blob/testset/black_test \
    --name ${NAME} \
    --phase ${PHASE} \
    --max_epsilon ${EPS} \
    --batchSize 1 \
    --iter ${Iter} \
    --confidence 5 \
    2>&1 | tee ./test/${NAME}.txt 

