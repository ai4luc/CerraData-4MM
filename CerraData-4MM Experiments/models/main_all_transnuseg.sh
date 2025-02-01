export CUDA_LAUNCH_BLOCKING=1
# SAR
export PYTHON_PATH_SAR_7W='/home/mateus/MateusPro/dlr_project/models/sar/transnuseg/main_tns_7sar_w.sh'
export PYTHON_PATH_SAR14='/home/mateus/MateusPro/dlr_project/models/sar/transnuseg/main_tns_14sar_nonw.sh'
export PYTHON_PATH_SAR14W='/home/mateus/MateusPro/dlr_project/models/sar/transnuseg/main_tns_14sar_w.sh'

# MSI
export PYTHON_PATH_MSI_7W='/home/mateus/MateusPro/dlr_project/models/msi/transnuseg/main_tns_7msi_w.sh'
export PYTHON_PATH_MSI14='/home/mateus/MateusPro/dlr_project/models/msi/transnuseg/main_tns_14msi_nonw.sh'
export PYTHON_PATH_MSI14W='/home/mateus/MateusPro/dlr_project/models/msi/transnuseg/main_tns_14msi_w.sh'
# Concat
export PYTHON_PATH_CONC_7W='/home/mateus/MateusPro/dlr_project/models/concat/transnuseg/main_tns_7conc_w.sh'
export PYTHON_PATH_CONC14='/home/mateus/MateusPro/dlr_project/models/concat/transnuseg/main_tns_14conc_nonw.sh'
export PYTHON_PATH_CONC14W='/home/mateus/MateusPro/dlr_project/models/concat/transnuseg/main_tns_14conc_w.sh'

# SAR
TORCH_USE_CUDA_DSA=1 bash $PYTHON_PATH_SAR_7W
TORCH_USE_CUDA_DSA=1 bash $PYTHON_PATH_SAR14
TORCH_USE_CUDA_DSA=1 bash $PYTHON_PATH_SAR14W

# MSI
TORCH_USE_CUDA_DSA=1 bash $PYTHON_PATH_MSI_7W
TORCH_USE_CUDA_DSA=1 bash $PYTHON_PATH_MSI14
TORCH_USE_CUDA_DSA=1 bash $PYTHON_PATH_MSI14W

# CONCAT
TORCH_USE_CUDA_DSA=1 bash $PYTHON_PATH_CONC_7W
TORCH_USE_CUDA_DSA=1 bash $PYTHON_PATH_CONC14
TORCH_USE_CUDA_DSA=1 bash $PYTHON_PATH_CONC14W


