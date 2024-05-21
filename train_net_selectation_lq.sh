#! /bin/bash

model=unet
batch=1
loss=dice2
reps='0,1,2,3,4'
assumption=slice
slices_per_volume=0

cuda=1

drop_p=0.0
incl_p=0.0
max_iter=6

for hq_percentage in 0.1 0.2 0.3
do
    for total_slices in 100
    do
        for n_volumes in 10 
        do
            for ds in granulocytes
            do
                for mode in segmentation_debiased_input
                do

                    python scripts/train_net_selection_lq.py $model $batch $loss $cuda $reps $drop_p $incl_p $max_iter $mode $assumption $n_volumes $ds $slices_per_volume $total_slices $hq_percentage > out/out_${ds}_${mode}_${assumption}_${loss}_${model}_${drop_p}_${incl_p}_${max_iter}_v${n_volumes}_total_slices_${total_slices}_hq_${hq_percentage}.txt 2> err/err_${ds}_${mode}_${assumption}_${loss}_${model}_${drop_p}_${incl_p}_${max_iter}_v${n_volumes}_total_slices_${total_slices}_hq_${hq_percentage}.txt &
                    wait
                done
            done
        done
    done
done