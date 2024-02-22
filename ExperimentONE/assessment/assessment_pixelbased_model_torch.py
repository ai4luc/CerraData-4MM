"""

"""

# -------- Library --------
# Machine Learning
import torch
import torch.nn as nn
from torch import Tensor

import segmentation_models_pytorch as smp

# Data
import cv2
import skimage.io as skio
import numpy as np
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms

# Graph
from typing import Any, Callable, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt

# Directory manager
import os
import glob
from copy import deepcopy
from pathlib import Path
import copy
import time

# Metrics
from sklearn.metrics import precision_score, f1_score

import argparse
from dataset import MyDatasetMTL, MyDatasetSTL
from utils import *
from transnuseg import TransNuSeg  # Criar uma outra função dentro da main que chama o modelo multi-task

# -------- Set up --------

# Constants
IMG_HEIGHT = 512
IMG_WIDTH = 512

POTSDAM_DATA_PATH = '../dataset/Potsdam_split_data2/test'

# On NVIDIA architecture
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# On Apple M chip architecture
# device = torch.device("mps")
print('Using ' + str(device) + ' device')


# --------------- Models ---------------
# DeepLabv3plus
def deeplabv3plus_model(num_channel: int, num_classes: int):
    deeplabv3plus = smp.DeepLabV3Plus(encoder_name='resnet50', encoder_weights=None,
                                      in_channels=num_channel, classes=num_classes, activation='softmax')

    return deeplabv3plus


# U-Net
def unet_model(num_channel: int, num_classes: int):
    unet = smp.Unet(encoder_name='resnet50', encoder_weights=None,
                    in_channels=num_channel, classes=num_classes,
                    activation='softmax')

    return unet


# --------------- Test Function ---------------
def main(path_trained_model, data, model_type: str):
    y_pred = []
    y_true = []

    if model_type == 'STL':
        # Model's architecture
        Tmodel = deeplabv3plus_model(num_channel=3, num_classes=6)

        # Trained model call
        Tmodel.load_state_dict(torch.load(path_trained_model))
        Tmodel.eval()

        Tmodel.to(device)
        print('STL Testing...')
        with torch.no_grad():
            # Data loading to GPU
            inputs = data.to(device)

            # Prediction
            y_pred = Tmodel(inputs)

            # GPU to CPU
            y_predmax = y_pred.argmax(1).cpu().numpy()

        return y_predmax

    elif model_type == 'MTL':
        Tmodel = TransNuSeg(img_size=512, in_chans=3)
        Tmodel.load_state_dict(torch.load(path_trained_model))
        Tmodel.to(device)
        Tmodel.eval()

        print('MTL Testing...')
        with (torch.no_grad()):
            # Data loading to GPU
            data = data.float()
            data = data.to(device)

            # Predicting
            y_pseg, y_pedg, y_pclu = Tmodel(data)

            # GPU to CPU
            y_pseg_max = y_pseg.argmax(1).cpu().numpy()
            y_pedg_max = y_pedg.argmax(1).cpu().numpy()
            y_pclu_max = y_pclu.argmax(1).cpu().numpy()

        return y_pseg_max, y_pedg_max, y_pclu_max

    else:
        print('Choose STL or MTL')


# --------- Single-task models ---------
# --- Data
data_path = POTSDAM_DATA_PATH
"""
test_set = MyDatasetSTL(dir_path=data_path)
testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)


# --- Deeplabv3
print('Saving DeepLabv3 predictions')
deeplabv3plus_path = '../models/deeplabv3plus/saved/modelDeepLab_epoch:2_testloss:1.3351634740829468_2024-02-17 15:10:59.031147.pt'
exp_directory = '/home/pedro/Desktop/experimentosMateus/Projeto/models/deeplabv3plus/outputs_150corr/cross_entropy/'
i = 0
for image, mask in testloader:
    deeplab_output = main(deeplabv3plus_path, data=image, model_type='STL')
    deeplab_output = np.array(deeplab_output, dtype='int32').reshape(512, 512)
    skio.imsave(exp_directory + 'deeplabv3_' + str(i) + '.tif', deeplab_output, check_contrast=False)
    i=i+1


# --- U-Net
print('Saving Unet predictions')
unet_path = '../models/unet/saved/modelDeepLab_epoch:24_testloss:1.2659084796905518_2024-02-18 10:51:28.255490.pt'
exp_directory = '/home/pedro/Desktop/experimentosMateus/Projeto/models/unet/outouts_150corr/cross_entropy/'
i=0
for image, mask in testloader:
    unet_output = main(unet_path, data=image, model_type='STL')
    unet_output = np.array(unet_output, dtype='int32').reshape(512, 512)
    skio.imsave(exp_directory + 'unet_' + str(i) + '.tif', unet_output, check_contrast=False)
    i=i+1
"""

# --- TransNuSeg
print('Saving TransNuSeg predictions')
test_set = MyDatasetMTL(dir_path=data_path)
testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

transnuseg_path = '../models/transnuseg/saved/model_epoch:147_testloss:0.5240399014565899_2024-02-15 13:41:19.455008.pt'
transnuseg_outputs = '../models/transnuseg/outputs_150corr/'
transnuseg_true = '/home/pedro/Desktop/experimentosMateus/Projeto/dataset/Potsdam_split_data2/test/'

i = 0
for img, instance_mask, semantic_mask, normal_edge_mask, cluster_edge_mask in testloader:
    y_pseg_max, y_pedg_max, y_pclu_max = main(transnuseg_path, data=img, model_type='MTL')

    # Predictions
    #y_pseg_max = np.array(y_pseg_max, dtype='int32').reshape(512,512)
    #skio.imsave(transnuseg_outputs + 'seg/' + 'transnuseg_pred_seg_' + str(i) + '.tif', y_pseg_max, check_contrast=False)
    # ----
    #y_pedg_max = np.array(y_pedg_max, dtype='int32').reshape(512, 512)
    #skio.imsave(transnuseg_outputs +'edg/'+ 'transnuseg_pred_edg_' + str(i) + '.tif', y_pedg_max, check_contrast=False)
    # ----
    #y_pclu_max = np.array(y_pclu_max, dtype='int32').reshape(512, 512)
    #skio.imsave(transnuseg_outputs + 'clu/' + 'transnuseg_pred_clu_' + str(i) + '.tif', y_pclu_max, check_contrast=False)

    # True
    #semantic_mask = semantic_mask.argmax(1).cpu().numpy()
    #skio.imsave(transnuseg_true + 'seg/' + 'seg_' + str(i) + '.tif', semantic_mask, check_contrast=False)
    # ----
    #normal_edge_mask = normal_edge_mask.argmax(1).cpu().numpy()
    #skio.imsave(transnuseg_true + 'edg/' + 'edg_' + str(i) + '.tif', normal_edge_mask, check_contrast=False)
    # ----
    cluster_edge_mask = cluster_edge_mask.argmax(1).cpu().numpy()
    skio.imsave(transnuseg_true + 'clu/' + 'clu_' + str(i) + '.tif', cluster_edge_mask, check_contrast=False)

    i=i+1



