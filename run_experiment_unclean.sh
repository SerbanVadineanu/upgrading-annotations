#! /bin/bash

mode=debiasing_input_unclean
err_type=omission
cuda_no=0
ds=epfl_train
model=unet


for n_volumes in 1
do
    for total_slices in 40
    do
        # Special experiment
        python scripts/run_experiment_unclean.py $mode $err_type $n_volumes $cuda_no $total_slices $ds $model > out_exp/out_unclean_${model}_${ds}_${mode}_${err_type}_${n_volumes}_total_slices_${total_slices}.txt 2> err_exp/err_unclean_${model}_${ds}_${mode}_${err_type}_${n_volumes}_total_slices_${total_slices}.txt &

        wait
    done
done

err_type=inclusion

for n_volumes in 1
do
    for total_slices in 80
    do
        # Special experiment
        python scripts/run_experiment_unclean.py $mode $err_type $n_volumes $cuda_no $total_slices $ds $model > out_exp/out_unclean_${model}_${ds}_${mode}_${err_type}_${n_volumes}_total_slices_${total_slices}.txt 2> err_exp/err_unclean_${model}_${ds}_${mode}_${err_type}_${n_volumes}_total_slices_${total_slices}.txt &

        wait
    done
done

err_type=bias

for n_volumes in 1
do
    for total_slices in 40
    do
        # Special experiment
        python scripts/run_experiment_unclean.py $mode $err_type $n_volumes $cuda_no $total_slices $ds $model > out_exp/out_unclean_${model}_${ds}_${mode}_${err_type}_${n_volumes}_total_slices_${total_slices}.txt 2> err_exp/err_unclean_${model}_${ds}_${mode}_${err_type}_${n_volumes}_total_slices_${total_slices}.txt &

        wait
    done
done