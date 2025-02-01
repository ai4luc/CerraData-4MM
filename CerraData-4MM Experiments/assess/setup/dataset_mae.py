import scipy.io as sio
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import glob
import cv2
from PIL import Image
from torchvision import transforms
import tifffile as tiff
from data_preprocessing import normalize, read_img

# On NVIDIA architecture
device = torch.device("mps")
print('Using ' + str(device) + ' device')

class MyDataset(Dataset):
    '''
    dir_path: path to data, having two folders named data and label respectively
    '''

    def __init__(self, dir_path, transform=None):
        self.dir_path = dir_path
        self.transform = transform
        # Data path
        self.data_path = os.path.join(dir_path, "opt_images")
        self.data_lists = sorted(glob.glob(os.path.join(self.data_path, "*.tif")))
        # Optical data
        self.opt_path = os.path.join(dir_path, "opt_images")
        self.opt_lists = sorted(glob.glob(os.path.join(self.opt_path, "*.tif")))
        # SAR data
        self.sar_path = os.path.join(dir_path, "sar_images")
        self.sar_lists = sorted(glob.glob(os.path.join(self.sar_path, "*.tif")))
        # Semantic mask
        self.label_path = os.path.join(dir_path, "semantic_label")
        self.label_lists = sorted(glob.glob(os.path.join(self.label_path, "*.tif")))
        # Edge mask
        self.edge_label_path = os.path.join(dir_path, "edge_label")
        self.edge_label_lists = sorted(glob.glob(os.path.join(self.edge_label_path, "*.tif")))

    def __getitem__(self, index):
        opt_img_path = self.opt_lists[index]
        sar_img_path = self.sar_lists[index]
        label_path = self.label_lists[index]
        edge_path = self.edge_label_lists[index]
        data_paths = self.data_lists[index]

        #opt_img = read_img(opt_img_path, dtype='opt')
        #sar_img = read_img(sar_img_path, dtype='sar')
        opt_img = read_img(opt_img_path)
        sar_img = read_img(sar_img_path)
        edge = tiff.imread(edge_path)
        semantic_mask = tiff.imread(label_path)
        normal_edge_mask = self.generate_normal_edge_mask(edge)

        if self.transform is not None:
            seed = 666
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            if torch.backends.mps.is_available():
                torch.mps.manual_seed(seed)

            opt_img = self.transform(opt_img)
            sar_img = self.transform(sar_img)
            semantic_mask = self.transform(semantic_mask)
            normal_edge_mask = self.transform(normal_edge_mask)
        else:
            opt_img = torch.tensor(opt_img, dtype=torch.float32)
            sar_img = torch.tensor(sar_img, dtype=torch.float32)
            semantic_mask = torch.tensor(semantic_mask, dtype=torch.int32)
            normal_edge_mask = torch.tensor(normal_edge_mask, dtype=torch.torch.int32)


        # Stack images along the channel dimension
        stacked_img = torch.cat((opt_img, sar_img), dim=0)
        ae_img = stacked_img # Use in Autoencoder task

        # To GPU
        ae_img = ae_img.to(device)
        stacked_img = stacked_img.to(device)
        semantic_mask = semantic_mask.to(device)
        normal_edge_mask = torch.tensor(normal_edge_mask, dtype=torch.int32).to(device)

        return data_paths, stacked_img, ae_img

    def __len__(self):
        return len(self.opt_lists)

    def sem2ins(self, label):
        #seg_mask_g = label.copy()
        contours, hierarchy = cv2.findContours(label, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i in range(len(contours)):
            cnt = contours[i]
            label = cv2.drawContours(label, [cnt], 0, i + 1, -1)
        return label

    def generate_normal_edge_mask(self, edge):
        normal_edge_mask = edge.copy()
        normal_edge_mask[normal_edge_mask == 0] = 0  # Non edge
        normal_edge_mask[normal_edge_mask == 255] = 1  # Edge
        return normal_edge_mask
