which_dataset=$1

if [ $which_dataset == coco ]; then
    exp_ID=GT_test_COCO_res50_256x192_advmix
    cfg_file=experiments/debug.yaml
    checkpoint=models/pytorch/pose_coco/pose_resnet_50_256x192.pth
elif [ $which_dataset == mpii ]; then
    exp_ID=GT_test_MPII_res50_256x256_advmix
    cfg_file=experiments/debug.yaml
    checkpoint=models/pytorch/pose_mpii/pose_resnet_50_256x256.pth.tar
fi
echo 'Start testing :'$which_dataset


python tools/test_corruption.py \
    --cfg  $cfg_file \
    --test_robust \
    --exp_id $exp_ID \
    --save_suffix $exp_ID \
    TEST.MODEL_FILE $checkpoint \
    TEST.USE_GT_BBOX True