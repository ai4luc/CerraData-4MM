export CUDA_LAUNCH_BLOCKING=1
export MODEL_TYPE='UNet'
export RANDOM_SEED=666
export BATCH_SIZE=2
export DATASET='Potsdam'
export NUM_CHANNEL=3
export NUM_CLASSES=6
export NUM_EPOCH=35
export LR=0.001
export PYTHON_PATH='./unet_torch.py'

TORCH_USE_CUDA_DSA=1 python $PYTHON_PATH --model_type=$MODEL_TYPE --random_seed=$RANDOM_SEED --batch_size=$BATCH_SIZE --dataset=$DATASET --num_channel=$NUM_CHANNEL --num_classes=$NUM_CLASSES --num_epoch=$NUM_EPOCH  --lr=$LR

