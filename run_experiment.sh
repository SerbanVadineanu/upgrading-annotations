#! /bin/bash

mode=debiasing_input
err_type=inclusion
cuda_no=1
ds=epfl
model=unet


for n_volumes in 1
do
    for total_slices in 2 5 10 20 40 80
    do
        # Special experiment
        python scripts/run_experiment_special.py $mode $err_type $n_volumes $cuda_no $total_slices $ds $model > out_exp/out_${model}_${ds}_${mode}_${err_type}_${n_volumes}_total_slices_${total_slices}.txt 2> err_exp/err_${model}_${ds}_${mode}_${err_type}_${n_volumes}_total_slices_${total_slices}.txt &

        wait
    done
done