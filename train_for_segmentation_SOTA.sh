#! /bin/bash

model=unet
batch=1
loss=dice2
reps='0,1,2,3,4'
assumption=slice
slices_per_volume=0

cuda=1

drop_p=0.0
incl_p=0.7
max_iter=0

n_volumes=10


# Partial SOTA
for mode in segmentation
do
    for ds in inclusion_hl60
    do
        for total_slices in 10
        do
            python scripts/train_net_sota_partial.py $model $batch $loss $cuda $reps $drop_p $incl_p $max_iter $mode $assumption $n_volumes $ds $slices_per_volume $total_slices > out_sota/out_partial_${ds}_${mode}_${assumption}_${loss}_${model}_${drop_p}_${incl_p}_${max_iter}_v${n_volumes}_total_slices_${total_slices}.txt 2> err_sota/err_partial_${ds}_${mode}_${assumption}_${loss}_${model}_${drop_p}_${incl_p}_${max_iter}_v${n_volumes}_total_slices_${total_slices}.txt &
            wait
        done
    done
done



# Confident SOTA
# for mode in segmentation
# do
#     for ds in inclusion_hl60
#     do
#         for total_slices in 10
#         do
#             python scripts/train_net_sota_confident.py $model $batch $loss $cuda $reps $drop_p $incl_p $max_iter $mode $assumption $n_volumes $ds $slices_per_volume $total_slices > out_sota/out_confident_${ds}_${mode}_${assumption}_${loss}_${model}_${drop_p}_${incl_p}_${max_iter}_v${n_volumes}_total_slices_${total_slices}.txt 2> err_sota/err_confident_${ds}_${mode}_${assumption}_${loss}_${model}_${drop_p}_${incl_p}_${max_iter}_v${n_volumes}_total_slices_${total_slices}.txt &
#             wait
#         done
#     done
# done
