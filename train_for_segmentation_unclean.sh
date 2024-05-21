#! /bin/bash

model=unet
batch=2
loss=dice2
reps='0,1,2,3,4'
assumption=slice
slices_per_volume=0

cuda=0

drop_p=0.0
incl_p=0.0
max_iter=0

n_volumes=10
# total_slices=10
ds=hl60

# # inclusion
# drop_p_debiasing=0.0
# incl_p_debiasing=0.7
# max_iter_debiasing=0



# for mode in segmentation_unclean
# do
#     for incl_p in 0.1 0.2 0.3 0.5 0.7
#     do
#         python scripts/train_net.py $model $batch $loss $cuda $reps $drop_p $incl_p $max_iter $mode $assumption $n_volumes $ds $slices_per_volume $total_slices $drop_p_debiasing $incl_p_debiasing $max_iter_debiasing > out/out_${ds}_${mode}_${assumption}_${loss}_${model}_${drop_p}_${incl_p}_${max_iter}_v${n_volumes}_total_slices_${total_slices}_${drop_p_debiasing}_${incl_p_debiasing}_${max_iter_debiasing}.txt 2> err/err_${ds}_${mode}_${assumption}_${loss}_${model}_${drop_p}_${incl_p}_${max_iter}_v${n_volumes}_total_slices_${total_slices}_${drop_p_debiasing}_${incl_p_debiasing}_${max_iter_debiasing}.txt &
#         wait
#     done
# done

# omission
# total_slices=80

# drop_p_debiasing=0.7
# incl_p_debiasing=0.0
# max_iter_debiasing=0

# drop_p=0.0
# incl_p=0.0
# max_iter=0


# for mode in segmentation_unclean
# do
#     for drop_p in 0.1 0.2 0.3 0.5 0.7
#     do
#         python scripts/train_net.py $model $batch $loss $cuda $reps $drop_p $incl_p $max_iter $mode $assumption $n_volumes $ds $slices_per_volume $total_slices $drop_p_debiasing $incl_p_debiasing $max_iter_debiasing > out/out_${ds}_${mode}_${assumption}_${loss}_${model}_${drop_p}_${incl_p}_${max_iter}_v${n_volumes}_total_slices_${total_slices}_${drop_p_debiasing}_${incl_p_debiasing}_${max_iter_debiasing}.txt 2> err/err_${ds}_${mode}_${assumption}_${loss}_${model}_${drop_p}_${incl_p}_${max_iter}_v${n_volumes}_total_slices_${total_slices}_${drop_p_debiasing}_${incl_p_debiasing}_${max_iter_debiasing}.txt &
#         wait
#     done
# done

# bias
total_slices=10

drop_p_debiasing=0.0
incl_p_debiasing=0.0
max_iter_debiasing=6

drop_p=0.0
incl_p=0.0
max_iter=0


for mode in segmentation_unclean
do
    for max_iter in 2 4 6
    do
        python scripts/train_net.py $model $batch $loss $cuda $reps $drop_p $incl_p $max_iter $mode $assumption $n_volumes $ds $slices_per_volume $total_slices $drop_p_debiasing $incl_p_debiasing $max_iter_debiasing > out/out_${ds}_${mode}_${assumption}_${loss}_${model}_${drop_p}_${incl_p}_${max_iter}_v${n_volumes}_total_slices_${total_slices}_${drop_p_debiasing}_${incl_p_debiasing}_${max_iter_debiasing}.txt 2> err/err_${ds}_${mode}_${assumption}_${loss}_${model}_${drop_p}_${incl_p}_${max_iter}_v${n_volumes}_total_slices_${total_slices}_${drop_p_debiasing}_${incl_p_debiasing}_${max_iter_debiasing}.txt &
        wait
    done
done