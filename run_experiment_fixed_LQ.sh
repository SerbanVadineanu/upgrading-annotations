#! /bin/bash

drop_p=0.0
incl_p=0.7
max_iter=0
dataset=inclusion_hl60
cuda_no=0

python scripts/run_experiment_segmentation_fixed_LQ.py $drop_p $incl_p $max_iter $dataset $cuda_no > out_exp/out_${dataset}_${drop_p}_${incl_p}_${max_iter}.txt 2> err_exp/err_${dataset}_${drop_p}_${incl_p}_${max_iter}.txt &