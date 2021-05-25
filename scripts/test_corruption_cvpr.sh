dataset=coco
allGPU=$1
node=$2


exp_ID=GT_test_MPII_HR32_256x256_kd_mse
# exp_ID=GT_test_MPII_res50_256x256_kd_mse
# exp_ID=GT_test_COCO_res50_256x192_mse


cfg_file=experiments/mpii/hrnet/w32_256x256_adam_lr1e-3_kd_mse.yaml
# cfg_file=experiments/mpii/resnet/res50_256x256_d256x3_adam_lr1e-3_kd_mse.yaml
# cfg_file=experiments/coco/resnet/res50_256x192_d256x3_adam_lr1e-3_kd_mse.yaml


checkpoint=output_cvpr/coco/pose_resnet/MPII_HR32_256x256_kd_mse/model_best_D.pth
# checkpoint=output_cvpr/coco/pose_resnet/MPII_res50_256x256_kd_mse/model_best_D.pth
# checkpoint=output_cvpr/coco/pose_resnet/COCO_res50_256x192_mse/model_best_D.pth


python tools/test_corruption.py \
    --cfg  $cfg_file \
    --test_robust \
    --exp_id $exp_ID \
    --save_suffix $exp_ID \
    TEST.MODEL_FILE $checkpoint \
    TEST.USE_GT_BBOX False