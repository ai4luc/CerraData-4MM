# -*- coding: utf-8 -*-
"""DeepLabv3plus_normalized0to1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17qCHgOV7M-0_gzwE4RPJ2NC361KbiG5j
"""

# -------- Library --------
# Machine Learning

import torch
import torch.nn as nn
from torch import optim
import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
import torch.nn.functional as F

# Data
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np

# Graph
from typing import Any, Callable, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt

# Directory manager
import os
import glob
from pathlib import Path
import copy
import time
from collections import Counter
from datetime import datetime
import logging
import sys

import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'util')))
from dataset_loader import MSIDataset
from utils import create_dir, draw_loss
from loss import DiceLoss

# -------- Set up --------

# Constants
IMG_HEIGHT = 128
IMG_WIDTH = 128

DATA_PATH = '/home/mateus/MateusPro/dlr_project/cerradata4m_exp/train/'

# On NVIDIA architecture
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# On Apple M chip architecture
# device = torch.device("mps")

print('Using ' + str(device) + ' GPU')


def save_training_history(train_loss, val_loss, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=700)
    plt.close()


# U-Net
def unet_model(num_channel: int, num_classes: int):
    model_unet = smp.Unet(encoder_name='resnet50', encoder_weights=None,
                          in_channels=num_channel, classes=num_classes, activation='softmax')
    return model_unet


# Training Function
def main():
    # Settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=True, default="UNet",
                        help="declare the model type to use, currently only support input being UNet")
    parser.add_argument("--random_seed", required=True, help="random seed")
    parser.add_argument("--batch_size", required=True, help="batch size")
    parser.add_argument("--num_channel", required=True, default=12, help="number of Channels")
    parser.add_argument("--num_classes", required=True, default=14, help="number of classes")
    parser.add_argument("--num_epoch", required=True, help='number of epoches')
    parser.add_argument("--lr", required=True, help="learning rate")
    parser.add_argument("--model_path", default=None, help="the path to the pretrained model")

    args = parser.parse_args()
    model_type = args.model_type
    num_channel = int(args.num_channel)
    num_classes = int(args.num_classes)
    batch_size = int(args.batch_size)
    random_seed = int(args.random_seed)
    num_epoch = int(args.num_epoch)
    base_lr = float(args.lr)

    # Report log
    now = datetime.now()
    create_dir('/home/mateus/MateusPro/dlr_project/models/msi/unet/log')
    logging.basicConfig(filename='/home/mateus/MateusPro/dlr_project/models/msi/unet/log/log_{}_{}.txt'.format(model_type, str(now)), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("Batch size : {} , epoch num: {}, num_channel: "
                 "{}, base_lr : {}".format(batch_size,num_epoch,num_channel,base_lr))


    # Data loader
    total_data = MSIDataset(dir_path=DATA_PATH, gpu=device, norm='none') # norm: "none", "0to1", "1to1"
    train_set_size = int(len(total_data) * 0.8)
    val_set_size = len(total_data) - train_set_size
    train_set, val_set = random_split(total_data, [train_set_size, val_set_size], generator=torch.Generator().manual_seed(random_seed))
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Report
    logging.info("train size {} val size {}".format(train_set_size, val_set_size))
    dataloaders = {"train": trainloader, "val": valloader}
    dataset_sizes = {"train": len(trainloader), "val": len(valloader)}
    logging.info("size train : {}, size val {} ".format(dataset_sizes["train"], dataset_sizes["val"]))

    val_loss = []
    train_loss = []
    all_loss = []

    # Weights
    class_counter = Counter()
    for _, labels, edge in trainloader:
        mask = labels.view(-1).cpu().numpy()
        class_counter.update(mask.tolist())

    total_pixels = sum(class_counter.values())
    class_frequency = {cls: count / total_pixels for cls, count in class_counter.items()}
    class_weights = {cls: 1.0 / freq for cls, freq in class_frequency.items()}
    max_weight = max(class_weights.values())
    class_weights = {cls: weight / max_weight for cls, weight in class_weights.items()}
    weight_tensor = torch.tensor([class_weights[i] for i in range(len(class_weights))], dtype=torch.float).to(device)

    print('Weight tensor: ', weight_tensor)

    # Model
    model = unet_model(num_channel=num_channel, num_classes=num_classes)
    best_model_wts = copy.deepcopy(model.state_dict())

    # Loss function
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    #criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=base_lr)

    best_loss = float('inf')
    best_epoch = 0

    model.to(device)

    for epoch in range(num_epoch):
        print(f'====== Epoch {epoch} ======')
        if epoch > best_epoch + 50:
            break
        for phase in ['train', 'val']:
            running_loss = 0.0
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for img, semantic_seg_mask, _ in dataloaders[phase]:
                img = img.float().to(device)
                semantic_seg_mask = semantic_seg_mask.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Model prediction
                    output = model(img)
                    # Loss
                    loss = criterion(output, semantic_seg_mask.long())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * img.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f}')
            logging.info(f'{phase} Loss: {epoch_loss:.4f}')

            # Saving the historical training
            if phase == 'train':
                train_loss.append(epoch_loss)
            else:
                val_loss.append(epoch_loss)

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, f'/home/mateus/MateusPro/dlr_project/models/msi/unet/saved_14/unet_14cMSI_W_model_epoch_{epoch}.pt')
                print(f'Best model nonWN 14c saved at epoch {epoch} with loss {best_loss:.4f}')

        save_training_history(train_loss, val_loss, f'/home/mateus/MateusPro/dlr_project/models/msi/unet/log/training_history_unet_14cMSI_W_{model_type}.png')
        torch.cuda.empty_cache()

        model.load_state_dict(best_model_wts)
        model.eval()

        dice_acc_val = 0
        dice_loss_val = DiceLoss(num_classes)

        with torch.no_grad():
            print('Validation...')
            for img, semantic_seg_mask, _ in valloader:
                img = img.float().to(device)
                semantic_seg_mask = semantic_seg_mask.to(device)
                output1 = model(img)
                print('Pred', np.unique(torch.argmax(output1, dim=1).data.cpu().numpy().ravel()))
                print('True', np.unique(semantic_seg_mask.data.cpu().numpy().ravel()))
                d_l = dice_loss_val(output1, semantic_seg_mask.float(), softmax=True)
                dice_acc_val += 1 - d_l.item()

        logging.info("dice_acc {}".format(dice_acc_val / dataset_sizes['val']))


if __name__ == '__main__':
    main()
    print('Fertig!')
