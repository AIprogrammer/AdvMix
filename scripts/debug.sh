allGPU=$1
node=$2

exp_ID=debug

cfg_file=experiments/mpii/hrnet/debug.yaml

checkpoint=models/pytorch/pose_mpii/pose_hrnet_w32_256x256.pth
# checkpoint=models/pytorch/pose_mpii/pose_resnet_50_256x256.pth.tar
# checkpoint=models/pytorch/pose_coco/pose_resnet_50_256x192.pth


python tools/train.py \
    --cfg  $cfg_file \
    --exp_id $exp_ID \
    --save_suffix $exp_ID \
    --load_from_D $checkpoint \
    --sample_times 1 \
    --joints_num 16 \
    --random_corruption \
    TEST.MODEL_FILE $checkpoint \
    TEST.USE_GT_BBOX True

# dataset=coco
# allGPU=$1
# node=$2


# exp_ID=MPII_HR32_256x256_kd_mse
# # exp_ID=MPII_res50_256x256_kd_mse

# cfg_file=experiments/mpii/hrnet/debug.yaml
# # cfg_file=experiments/mpii/resnet/res50_256x256_d256x3_adam_lr1e-3_kd_mse.yaml

# checkpoint=models/pytorch/pose_mpii/pose_hrnet_w32_256x256.pth
# # checkpoint=models/pytorch/pose_mpii/pose_resnet_50_256x256.pth.tar


# python tools/test_corruption.py \
#     --cfg  $cfg_file \
#     --test_robust \
#     --exp_id $exp_ID \
#     --save_suffix $exp_ID \
#     TEST.MODEL_FILE $checkpoint \
#     TEST.USE_GT_BBOX False
# 2>&1|tee -a $logdir

