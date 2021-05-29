which_dataset=$1

if [ $which_dataset == coco ]; then
    exp_ID=GT_test_COCO_res50_256x192_advmix
    cfg_file=experiments/coco/resnet/res50_256x192_d256x3_adam_lr1e-3_advmix.yaml
    checkpoint=output/coco/pose_resnet/MPII_res50_256x256_advmix/model_best_D.pth
elif [ $which_dataset == mpii ]; then
    exp_ID=GT_test_MPII_res50_256x256_advmix
    cfg_file=experiments/mpii/resnet/res50_256x256_d256x3_adam_lr1e-3_advmix.yaml
    checkpoint=output/coco/pose_resnet/COCO_res50_256x192_advmix/model_best_D.pth
fi


python tools/test_corruption.py \
    --cfg  $cfg_file \
    --test_robust \
    --exp_id $exp_ID \
    --save_suffix $exp_ID \
    TEST.MODEL_FILE $checkpoint \
    TEST.USE_GT_BBOX False