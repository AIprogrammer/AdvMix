which_dataset=$1

if [ $which_dataset == coco ]; then
    num_joints=17
    exp_ID=COCO_res50_256x192_kd_mse
    cfg_file=experiments/coco/resnet/res50_256x192_d256x3_adam_lr1e-3_kd_mse.yaml
    checkpoint=models/pytorch/pose_coco/pose_resnet_50_256x192.pth
elif [ $which_dataset == mpii ]; then
    num_joints=16
    exp_ID=MPII_res50_256x256_kd_mse
    cfg_file=experiments/mpii/resnet/res50_256x256_d256x3_adam_lr1e-3_kd_mse.yaml
    checkpoint=models/pytorch/pose_mpii/pose_resnet_50_256x256.pth.tar
fi
echo 'Start training :'$which_dataset

python tools/train.py \
    --cfg  $cfg_file \
    --exp_id $exp_ID \
    --save_suffix $exp_ID \
    --load_from_D $checkpoint \
    --advmix \
    --sample_times 3 \
    --joints_num $num_joints \
    --kd \
    --kd_mseloss \
    --alpha 0.5 \
    TEST.MODEL_FILE $checkpoint \
    TEST.USE_GT_BBOX True