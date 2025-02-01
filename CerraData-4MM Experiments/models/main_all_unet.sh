export CUDA_LAUNCH_BLOCKING=1
# 7 classes
# SAR
export PYTHON_PATH_USAR_7W='/home/mateus/MateusPro/dlr_project/models/sar/unet/main_u_sar7W.sh'
export PYTHON_PATH_USAR7='/home/mateus/MateusPro/dlr_project/models/sar/unet/main_u_sar_7nonw.sh'
# MSI
export PYTHON_PATH_UMSI_7W='/home/mateus/MateusPro/dlr_project/models/msi/unet/main_u_msi7W.sh'
export PYTHON_PATH_UMSI7='/home/mateus/MateusPro/dlr_project/models/msi/unet/main_u_msi_7nonw.sh'
# Concat
export PYTHON_PATH_UCONC_7W='/home/mateus/MateusPro/dlr_project/models/concat/unet/main_u_concat7W.sh'
export PYTHON_PATH_UCONC7='/home/mateus/MateusPro/dlr_project/models/concat/unet/main_u_concat_7nonw.sh'


# 14 classes
#export PYTHON_PATH_USAR_14W='/home/mateus/MateusPro/dlr_project/models/sar/unet/'
#export PYTHON_PATH_USAR14='/home/mateus/MateusPro/dlr_project/models/sar/unet/'
# MSI
#export PYTHON_PATH_UMSI_14W='/home/mateus/MateusPro/dlr_project/models/msi/unet/'
export PYTHON_PATH_UMSI14='/home/mateus/MateusPro/dlr_project/models/msi/unet/main_u_msi_nonw.sh'
# Concat
#export PYTHON_PATH_UCONC_14W='/home/mateus/MateusPro/dlr_project/models/concat/unet/main_u_concatW.sh'
export PYTHON_PATH_UCONC14='/home/mateus/MateusPro/dlr_project/models/concat/unet/main_u_concat_nonw.sh'

# Call Transnuseg exe
export TRANSNUSEG_EXE='/home/mateus/MateusPro/dlr_project/models/main_all_transnuseg.sh'


# 7 classes
TORCH_USE_CUDA_DSA=1 bash $PYTHON_PATH_USAR_7W
TORCH_USE_CUDA_DSA=1 bash $PYTHON_PATH_USAR7

TORCH_USE_CUDA_DSA=1 bash $PYTHON_PATH_UMSI_7W
TORCH_USE_CUDA_DSA=1 bash $PYTHON_PATH_UMSI7

TORCH_USE_CUDA_DSA=1 bash $PYTHON_PATH_UCONC_7W
TORCH_USE_CUDA_DSA=1 bash $PYTHON_PATH_UCONC7

# 14 classes
#TORCH_USE_CUDA_DSA=1 bash $PYTHON_PATH_USAR_14W
#TORCH_USE_CUDA_DSA=1 bash $PYTHON_PATH_USAR14

#TORCH_USE_CUDA_DSA=1 bash $PYTHON_PATH_UMSI_14W
TORCH_USE_CUDA_DSA=1 bash $PYTHON_PATH_UMSI14

#TORCH_USE_CUDA_DSA=1 bash $PYTHON_PATH_UCONC_14W
TORCH_USE_CUDA_DSA=1 bash $PYTHON_PATH_UCONC14

# Tranuseg
TORCH_USE_CUDA_DSA=1 bash $TRANSNUSEG_EXE

