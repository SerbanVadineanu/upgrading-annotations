#! /bin/bash

model=unet
batch=1
loss=dice2
reps='0,1,2,3,4'
assumption=slice
slices_per_volume=0

cuda=0

drop_p=0.3
incl_p=0.3
max_iter=4

n_volumes=10



for mode in segmentation_debiased_input
do
    for ds in inclusion_granulocytes
    do
        for total_slices in 10 40 80
        do

            python scripts/train_net.py $model $batch $loss $cuda $reps $drop_p $incl_p $max_iter $mode $assumption $n_volumes $ds $slices_per_volume $total_slices $total_slices_biased > out/out_${ds}_${mode}_${assumption}_${loss}_${model}_${drop_p}_${incl_p}_${max_iter}_v${n_volumes}_total_slices_${total_slices}_biased_slices_${total_slices_biased}.txt 2> err/err_${ds}_${mode}_${assumption}_${loss}_${model}_${drop_p}_${incl_p}_${max_iter}_v${n_volumes}_total_slices_${total_slices}_biased_slices_${total_slices_biased}.txt &
            wait
        done
    done
done






#LQ training
# drop_p=0.7
# incl_p=0.0
# max_iter=0

# total_slices=0

# for mode in segmentation_debiased_input
# do
#     for ds in inclusion_hl60
#     do
#         for total_slices in 10 20 40 80 160
#         do
#             for total_slices_biased in 10 20 40 80 160
#             do
#                 python scripts/train_net.py $model $batch $loss $cuda $reps $drop_p $incl_p $max_iter $mode $assumption $n_volumes $ds $slices_per_volume $total_slices $total_slices_biased > out/out_${ds}_${mode}_${assumption}_${loss}_${model}_${drop_p}_${incl_p}_${max_iter}_v${n_volumes}_total_slices_${total_slices}_biased_slices_${total_slices_biased}.txt 2> err/err_${ds}_${mode}_${assumption}_${loss}_${model}_${drop_p}_${incl_p}_${max_iter}_v${n_volumes}_total_slices_${total_slices}_biased_slices_${total_slices_biased}.txt &
#                 wait
#             done
#         done
#     done
# done


# drop_p=0.0
# incl_p=0.7
# max_iter=0

# total_slices=0

# for mode in segmentation_debiased_input
# do
#     for ds in inclusion_hl60
#     do
#         for total_slices in 10 20 40 80 160
#         do
#             for total_slices_biased in 10 20 40 80 160
#             do
#                 python scripts/train_net.py $model $batch $loss $cuda $reps $drop_p $incl_p $max_iter $mode $assumption $n_volumes $ds $slices_per_volume $total_slices $total_slices_biased > out/out_${ds}_${mode}_${assumption}_${loss}_${model}_${drop_p}_${incl_p}_${max_iter}_v${n_volumes}_total_slices_${total_slices}_biased_slices_${total_slices_biased}.txt 2> err/err_${ds}_${mode}_${assumption}_${loss}_${model}_${drop_p}_${incl_p}_${max_iter}_v${n_volumes}_total_slices_${total_slices}_biased_slices_${total_slices_biased}.txt &
#                 wait
#             done
#         done
#     done
# done


# drop_p=0.0
# incl_p=0.0
# max_iter=6

# total_slices=0

# for mode in segmentation_debiased_input
# do
#     for ds in inclusion_hl60
#     do
#         for total_slices in 10 20 40 80 160
#         do
#             for total_slices_biased in 10 20 40 80 160
#             do
#                 python scripts/train_net.py $model $batch $loss $cuda $reps $drop_p $incl_p $max_iter $mode $assumption $n_volumes $ds $slices_per_volume $total_slices $total_slices_biased > out/out_${ds}_${mode}_${assumption}_${loss}_${model}_${drop_p}_${incl_p}_${max_iter}_v${n_volumes}_total_slices_${total_slices}_biased_slices_${total_slices_biased}.txt 2> err/err_${ds}_${mode}_${assumption}_${loss}_${model}_${drop_p}_${incl_p}_${max_iter}_v${n_volumes}_total_slices_${total_slices}_biased_slices_${total_slices_biased}.txt &
#                 wait
#             done
#         done
#     done
# done