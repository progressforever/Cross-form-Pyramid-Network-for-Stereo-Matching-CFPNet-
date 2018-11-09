#!/bin/bash
#only for testing
python flyingsense_train.py --maxdisp 192 \
               --model stackhourglass\
               --datapath /home1/Documents/Database/ \
               --epochs 0 \
               --loadmodel ./train_model/flyingthings_factorization_4407488_epoch_40_model/checkpoint_30.tar \
               #--savemodel ./train_model/flyingthings_final_4688514_epoch_40_model/ > ./runLogFlying/cfp/flyingthings_final_4688514_begin_epoch25_end_epoch_30_Run.log 2>&1 & 
               
               #--savemodel ./Data_occ_epoch_1175_Models/ > Data_occ_epoch_1175_Run.log 2>&1 &
