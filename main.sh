#!/bin/bash
#only for testing
nohup python main.py --maxdisp 192 \
               --model stackhourglass\
               --datapath /home1/Documents/Database/Kitti/training/ \
               --epochs 380 \
	       	   --loadmodel ./train_model/Data_2015_4688514_final_flying21_epoch_600_model/checkpoint_220.tar \
               --savemodel ./train_model/Data_2015_4688514_final_flying21_epoch_600_model/ > ./runLog2015/cfp/Data_2015_4688514_final_flying21_begin220_epoch_600_Run.log 2>&1 &
               #--loadmodel ./train_model/flyingthings_final_4688514_epoch_40_model/checkpoint_22.tar 
               
               
               #--savemodel ./train_model/Data_final_psm2015_50_model/ > Data_final_psm2015_50_Run.log 2>&1 &
               #> Data_final_occ_epoch_2000_Run.log 2>&1 &
               #--savemodel ./Data_occ_epoch_1175_Models/ > Data_occ_epoch_1175_Run.log 2>&1 &
