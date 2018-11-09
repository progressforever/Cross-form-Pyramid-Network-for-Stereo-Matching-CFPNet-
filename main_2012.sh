#!/bin/bash
#only for testing
nohup python main.py --maxdisp 192 \
               --model stackhourglass\
               --datapath /home1/Documents/Database/Kitti2012/training/ \
               --epochs 600 \
               --loadmodel ./train_model/flyingthings_final_4688514_epoch_40_model/checkpoint_22.tar \
               --savemodel ./train_model/Data_2012_4688514_final_flying22_epoch_600_model/ > ./runLog2012/cfp/Data_2012_4688514_final_flying22_epoch_600_Run.log 2>&1 &
