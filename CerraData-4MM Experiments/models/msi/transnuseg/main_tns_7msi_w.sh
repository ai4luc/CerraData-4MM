export CUDA_LAUNCH_BLOCKING=1
export PYTHON_PATH='/home/mateus/MateusPro/dlr_project/models/msi/transnuseg/train_7msi_w.py'
export MODEL_TYPE='transnuseg'
export ALPHA='0.3'
export BETA='0.35'
export GAMMA='0.35'
export SHARING_RATIO='0.8'
export NUM_EPOCH=100
export LR=0.001
export RANDOM_SEED=666
export BATCH_SIZE=32
export NUM_CHANNEL=12
export NUM_CLASSES=7

TORCH_USE_CUDA_DSA=1 python $PYTHON_PATH --model_type=$MODEL_TYPE --alpha=$ALPHA --beta=$BETA --gamma=$GAMMA --sharing_ratio=$SHARING_RATIO --lr=$LR --num_epoch=$NUM_EPOCH --random_seed=$RANDOM_SEED --batch_size=$BATCH_SIZE --num_channel=$NUM_CHANNEL --num_classes=$NUM_CLASSES
