#!/bin/bash
#only for testing
nohup python main.py --maxdisp 192 \
               --model stackhourglass\
               --datapath /home1/Documents/Database/Kitti2012/ \
               --epochs 500 \
               --loadmodel ./train_model/cfp/Data_2012_begin_epoch390_finutune_001_500_model/checkpoint_315.tar \
               --savemodel ./train_model/cfp/Data_2012_begin_epoch390_finutune_001_500_model/ > ./runLog2015/cfp/Data_2012_begin_epoch390_finutune_0001_begin_315_500_Run.log 2>&1 &
               #--loadmodel ./train_model/Data_2015_6164544_begin_test268_fintune_001_epoch_500_model/checkpoint_115.tar \
               
               
               #--savemodel ./train_model/Data_final_psm2015_50_model/ > Data_final_psm2015_50_Run.log 2>&1 &
               #> Data_final_occ_epoch_2000_Run.log 2>&1 &
               #--savemodel ./Data_occ_epoch_1175_Models/ > Data_occ_epoch_1175_Run.log 2>&1 &
