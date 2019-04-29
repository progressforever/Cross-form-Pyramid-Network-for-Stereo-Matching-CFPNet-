#!/bin/bash
#only for testing
nohup python main.py --maxdisp 192 \
               --model stackhourglass\
               --datapath /home1/Documents/Database/Kitti2012/training/ \
               --epochs 600 \
               --savemodel ./models > ./Run.log 2>&1 &
