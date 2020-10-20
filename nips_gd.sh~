#!/bin/bash

Iter=10000
EPS=$2
PHASE=TEST
now=$(date +"%Y%m%d_%H%M%S")
NAME=GreedyFool_E_${EPS}-Iter_${Iter}_$now
CUDA_VISIBLE_DEVICES=$1 python -u nips_gd.py \
    --dataroot /mnt/blob/testset/cifar_hard \
    --name ${NAME} \
    --phase ${PHASE} \
    --max_epsilon ${EPS} \
    --batchSize 1 \
    --iter ${Iter} \
    2>&1 | tee ./test/${NAME}.txt 
