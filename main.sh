#!/bin/bash
#only for testing
nohup python main.py --maxdisp 192 \
               --model stackhourglass\
               --datapath /home1/Documents/Database/Kitti/training/ \
               --epochs 1000 \
               --savemodel ./models/ > ./Run.log 2>&1 &

