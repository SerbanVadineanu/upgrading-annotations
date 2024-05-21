#! /bin/bash

model=unet
batch=1
loss=dice2
reps='0,1,2,3,4'
assumption=slice
slices_per_volume=0
mode=debiasing_input

cuda=1

drop_p=0.3
incl_p=0.3
max_iter=4


for total_slices in 10 40 80
do
    for n_volumes in 10
    do
        for ds in inclusion_granulocytes
        do
            for mode in debiasing_input
            do

                python scripts/train_net.py $model $batch $loss $cuda $reps $drop_p $incl_p $max_iter $mode $assumption $n_volumes $ds $slices_per_volume $total_slices > out/out_${ds}_${mode}_${assumption}_${loss}_${model}_${drop_p}_${incl_p}_${max_iter}_v${n_volumes}_total_slices_${total_slices}.txt 2> err/err_${ds}_${mode}_${assumption}_${loss}_${model}_${drop_p}_${incl_p}_${max_iter}_v${n_volumes}_total_slices_${total_slices}.txt &
                wait
            done
        done
    done
done


# drop_p=0.3
# incl_p=0.3
# max_iter=4

# for total_slices in 10 40 80
# do
#     for n_volumes in 10
#     do
#         for ds in inclusion_hl60
#         do
#             for mode in debiasing_input
#             do

#                 python scripts/train_net.py $model $batch $loss $cuda $reps $drop_p $incl_p $max_iter $mode $assumption $n_volumes $ds $slices_per_volume $total_slices > out/out_${ds}_${mode}_${assumption}_${loss}_${model}_${drop_p}_${incl_p}_${max_iter}_v${n_volumes}_total_slices_${total_slices}.txt 2> err/err_${ds}_${mode}_${assumption}_${loss}_${model}_${drop_p}_${incl_p}_${max_iter}_v${n_volumes}_total_slices_${total_slices}.txt &
#                 wait
#             done
#         done
#     done
# done