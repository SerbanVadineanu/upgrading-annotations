#! /bin/bash

model=unet
batch=2
loss=dice2
reps='0,1,2,3,4'
assumption=slice
slices_per_volume=0
mode=debiasing_input_unclean

cuda=0


for n_volumes in 1
do
    for total_slices in 10
    do
        for mode in debiasing_input_unclean
        do
            total_slices=40
            drop_p_debiasing=0.7
            incl_p_debiasing=0.0
            max_iter_debiasing=0
            for drop_p in 0.1 0.2 0.3
            do
                ds=epfl_train
                incl_p=0.0
                max_iter=0
                python scripts/train_net.py $model $batch $loss $cuda $reps $drop_p $incl_p $max_iter $mode $assumption $n_volumes $ds $slices_per_volume $total_slices $drop_p_debiasing $incl_p_debiasing $max_iter_debiasing > out/out_${ds}_${mode}_${assumption}_${loss}_${model}_${drop_p}_${incl_p}_${max_iter}_v${n_volumes}_total_slices_${total_slices}_unclean_${drop_p_debiasing}_${incl_p_debiasing}_${max_iter_debiasing}.txt 2> err/err_${ds}_${mode}_${assumption}_${loss}_${model}_${drop_p}_${incl_p}_${max_iter}_v${n_volumes}_total_slices_${total_slices}_unclean_${drop_p_debiasing}_${incl_p_debiasing}_${max_iter_debiasing}.txt &
                wait
            done 
            total_slices=80
            drop_p_debiasing=0.0
            incl_p_debiasing=0.7
            max_iter_debiasing=0
            for incl_p in 0.1 0.2 0.3
            do
                ds=epfl_train
                max_iter=0
                drop_p=0.0
                python scripts/train_net.py $model $batch $loss $cuda $reps $drop_p $incl_p $max_iter $mode $assumption $n_volumes $ds $slices_per_volume $total_slices $drop_p_debiasing $incl_p_debiasing $max_iter_debiasing > out/out_${ds}_${mode}_${assumption}_${loss}_${model}_${drop_p}_${incl_p}_${max_iter}_v${n_volumes}_total_slices_${total_slices}_unclean_${drop_p_debiasing}_${incl_p_debiasing}_${max_iter_debiasing}.txt 2> err/err_${ds}_${mode}_${assumption}_${loss}_${model}_${drop_p}_${incl_p}_${max_iter}_v${n_volumes}_total_slices_${total_slices}_unclean_${drop_p_debiasing}_${incl_p_debiasing}_${max_iter_debiasing}.txt &
                wait
            done 
            total_slices=40
            drop_p_debiasing=0.0
            incl_p_debiasing=0.0
            max_iter_debiasing=6
            for max_iter in 2 4 6
            do
                ds=epfl_train
                incl_p=0.0
                drop_p=0.0
                python scripts/train_net.py $model $batch $loss $cuda $reps $drop_p $incl_p $max_iter $mode $assumption $n_volumes $ds $slices_per_volume $total_slices $drop_p_debiasing $incl_p_debiasing $max_iter_debiasing > out/out_${ds}_${mode}_${assumption}_${loss}_${model}_${drop_p}_${incl_p}_${max_iter}_v${n_volumes}_total_slices_${total_slices}_unclean_${drop_p_debiasing}_${incl_p_debiasing}_${max_iter_debiasing}.txt 2> err/err_${ds}_${mode}_${assumption}_${loss}_${model}_${drop_p}_${incl_p}_${max_iter}_v${n_volumes}_total_slices_${total_slices}_unclean_${drop_p_debiasing}_${incl_p_debiasing}_${max_iter_debiasing}.txt &
                wait
            done
        done
    done

done