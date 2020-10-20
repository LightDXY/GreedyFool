#!/bin/bash

Iter=10000
EPS=255
PHASE=BLACK
now=$(date +"%Y%m%d_%H%M%S")
NAME=GreedyFool_Black_E_${EPS}-Iter_${Iter}_$now
CUDA_VISIBLE_DEVICES=0 python -u nips_black_gd.py \
    --dataroot path/to/imgs/ \
    --name ${NAME} \
    --phase ${PHASE} \
    --max_epsilon ${EPS} \
    --batchSize 1 \
    --iter ${Iter} \
    --confidence 5 \
    2>&1 | tee ./test/${NAME}.txt 

