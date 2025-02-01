"""

"""
import numpy as np
# -------- Library --------
# Machine Learning

import segmentation_models_pytorch as smp

# Data
import skimage.io as skio
from torch.utils.data import DataLoader
import rasterio
from rasterio.merge import merge

# Metrics
import os
from utils import *
from transnuseg import TransNuSeg  # Criar uma outra funcao dentro da main que chama o modelo multi-task
from dataset4test import SARDataset, MSIDataset, MMDataset

# -------- Set up --------

# Constants
IMG_HEIGHT = 128
IMG_WIDTH = 128
nclasses = 14

# On NVIDIA architecture
#device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'mps'
print('Using ' + str(device) + ' device')


# --------------- U-Net architecture ---------------
def unet_model(num_channel: int, num_classes: int):
    unet = smp.Unet(encoder_name='resnet50', encoder_weights=None,
                    in_channels=num_channel, classes=num_classes,
                    activation='softmax')

    return unet


# --------------- Test Function ---------------
def model(path_trained_model, data, nchannel, model_type: str):
    y_pred = []
    y_true = []

    if model_type == 'STL':
        # Model's architecture
        Tmodel = unet_model(num_channel=nchannel, num_classes=nclasses)

        # Trained model call
        Tmodel.load_state_dict(torch.load(path_trained_model, map_location=device))
        Tmodel.eval()

        Tmodel.to(device)
        print('U-Net Testing...')
        with torch.no_grad():
            # Data loading to GPU
            inputs = data.to(device)

            # Prediction
            y_pred = Tmodel(inputs)

            # GPU to CPU
            y_predmax = y_pred.argmax(1).cpu().numpy()

        return y_predmax

    elif model_type == 'MTL':
        Tmodel = TransNuSeg(img_size=IMG_HEIGHT, in_chans=nchannel)
        Tmodel.load_state_dict(torch.load(path_trained_model, map_location=device))
        Tmodel.to(device)
        Tmodel.eval()

        print('MTL Testing...')
        with (torch.no_grad()):
            # Data loading to GPU
            data = data.float()
            data = data.to(device)

            # Predicting
            y_pseg, y_pedg = Tmodel(data)

            # GPU to CPU
            y_pseg_max = y_pseg.argmax(1).cpu().numpy()
            y_pedg_max = y_pedg.argmax(1).cpu().numpy()

        return y_pseg_max, y_pedg_max

    else:
        print('Choose STL or MTL')


def prediction(model_type, model_path, nchannel, data, save_pred_path):
    if model_type == 'MTL':
        for image, semantic, edge, datapath in data:
            # File information
            patch_name = os.path.basename(datapath[0])
            patch_ids = patch_name.split('_')[3].split('.tif')[0]
            print(patch_ids)

            # Get Geo information from the patch
            geo_img = rasterio.open(datapath[0])
            mosaic, out_trans = merge([geo_img])
            out_meta = geo_img.meta.copy()
            #crs = rasterio.crs.CRS({"init": "epsg:4326"})
            crs = geo_img.crs
            #print(f'Geo: CRS {crs}')

            # Model's prediction
            y_pseg_max, y_pedg_max = model(model_path, data=image, nchannel=nchannel, model_type=model_type)
            print(np.shape(y_pseg_max))

            # 1) Semantic Segmentation task prediction
            y_pseg_max = np.array(y_pseg_max, dtype='int32').reshape(IMG_HEIGHT, IMG_WIDTH)

            # Saving
            #print(f'TransNuSeg predictions: {np.unique(y_pseg_max)}')
            save_path = save_pred_path + 'seg/' + 'parrot_beak_transnuseg_' + str(patch_ids) + '.tif'

            with rasterio.open(save_path,
                               mode="w", driver='GTiff',
                               height=IMG_HEIGHT, width=IMG_WIDTH,
                               count=1, dtype='int32',
                               crs=crs, transform=out_trans
                               ) as dest:
                dest.write(y_pseg_max, 1)

            # 2) Edge Segmentation task prediction
            y_pedg_max = np.array(y_pedg_max, dtype='int32').reshape(IMG_HEIGHT, IMG_WIDTH)

            # Saving
            save_path_ed = save_pred_path + 'edg/' + 'parrot_beak_transnuseg_' + str(patch_ids) + '.tif'
            with rasterio.open(save_path_ed,
                               mode="w", driver='GTiff',
                               height=IMG_HEIGHT, width=IMG_WIDTH,
                               count=1, dtype='int32',
                               crs=crs, transform=out_trans
                               ) as dest:
                dest.write(y_pedg_max, 1)

    elif model_type == 'STL':
        for image, semantic, edge, datapath in data:
            # File information
            patch_name = os.path.basename(datapath[0])
            patch_ids = patch_name.split('_')[3].split('.tif')[0]
            print(patch_ids)

            # Get Geo information from the patch
            geo_img = rasterio.open(datapath[0])
            mosaic, out_trans = merge([geo_img])
            out_meta = geo_img.meta.copy()
            #crs = rasterio.crs.CRS({"init": "epsg:4326"})
            crs = geo_img.crs
            #print(f'Geo: CRS {crs}')

            # Saving the model's prediction
            unet_output = model(model_path, data=image, nchannel=nchannel, model_type=model_type)
            # To array
            unet_output = np.array(unet_output, dtype='int32').reshape(IMG_HEIGHT, IMG_WIDTH)
            #print(f'U-Net predictions: {np.unique(unet_output)}')

            # Saving
            save_path = save_pred_path + 'parrot_beak_unet_w_' + str(patch_ids) + '.tif'
            with rasterio.open(save_path,
                               mode="w", driver='GTiff',
                               height=128, width=128,
                               count=1, dtype='int32',
                               crs=crs, transform=out_trans
                               ) as dest:
                dest.write(unet_output, 1)
    else:
        print('Choose either "MTL" or "STL".')


# Data path
DATA_PATH = '../datasets/cerradata4mm_exp/test/'

# ####### SAR #######
# Data
test_sar = SARDataset(dir_path=DATA_PATH, gpu=device, norm='none')  #Norm: "none" "1to1" "0to1"
test_sar_load = torch.utils.data.DataLoader(test_sar, batch_size=1, shuffle=False)

# Model loading
sar_model = '../models/sar/unet/saved_14/unet_14cSAR_W_model_epoch_53.pt'
tns_predicition_save = '../models/sar/unet/pred_14/w/'

print('TransNuSeg for SAR online')

# Saving predictions
prediction(model_type='STL', model_path=sar_model, nchannel=2, data=test_sar_load,save_pred_path=tns_predicition_save)

# ####### MSI #######
# Data
test_msi = MSIDataset(dir_path=DATA_PATH, gpu=device, norm='none')  #Norm: "none" "1to1" "0to1"
test_msi_load = torch.utils.data.DataLoader(test_msi, batch_size=1, shuffle=False)

# TransNuSeg loading
msi_model = '../models/msi/unet/saved_14/unet_14cMSI_W_model_epoch_17.pt'
tns_MSIpred_save = '../models/msi/unet/pred_14/w/'

print('TransNuSeg for MSI online')

# Saving predictions
prediction(model_type='STL', model_path=msi_model, nchannel=12, data=test_msi_load,save_pred_path=tns_MSIpred_save)

# ####### MSI+SAR #######
# Data
test_msisar = MMDataset(dir_path=DATA_PATH, gpu=device, norm='none')  #Norm: "none" "1to1" "0to1"
test_msisar_load = torch.utils.data.DataLoader(test_msisar, batch_size=1, shuffle=False)

# TransNuSeg loading
tns_msisar_model = '../models/concat/unet/saved_14/unet_14cMSISAR_W_model_epoch_92.pt'
tns_MSISARpred_save = '../models/concat/unet/pred_14/w/'

print('TransNuSeg for MSI+SAR online')

# Saving predictions
prediction(model_type='STL', model_path=tns_msisar_model, nchannel=14, data=test_msisar_load,
           save_pred_path=tns_MSISARpred_save)
