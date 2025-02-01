export CUDA_LAUNCH_BLOCKING=1
export MODEL_TYPE='UNet'
export RANDOM_SEED=666
export BATCH_SIZE=32
export NUM_CHANNEL=14
export NUM_CLASSES=14
export NUM_EPOCH=100
export LR=0.001
export PYTHON_PATH='/home/mateus/MateusPro/dlr_project/models/concat/unet/unet_concW.py'

TORCH_USE_CUDA_DSA=1 python $PYTHON_PATH --model_type=$MODEL_TYPE --random_seed=$RANDOM_SEED --batch_size=$BATCH_SIZE --num_channel=$NUM_CHANNEL --num_classes=$NUM_CLASSES --num_epoch=$NUM_EPOCH  --lr=$LR

