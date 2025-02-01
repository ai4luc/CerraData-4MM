import scipy.io as sio
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import glob
import cv2
import tifffile as tiff
from osgeo import gdal
from torchvision import transforms

# On NVIDIA architecture
# device = torch.device("cuda")
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# On Apple M chip architecture
device = torch.device("mps")

print('Using ' + str(device) + ' device')


def read_img(path):
    sar_img = gdal.Open(path)
    sar_i = sar_img.ReadAsArray()
    return sar_i


def normalize(file_path):
    # Image reading
    img = read_img(file_path)
    img = np.clip(img, a_min=1e-6, a_max=None)  # Avoid log(0)

    # log
    log_img = np.log(img)

    # Mean and Std Dev (replace with actual values from previous calculation)
    img_mean = np.array([5.0567, 4.4802])
    img_std_dev = np.array([0.4312, 0.4301])

    # Normalization
    normalized = (log_img - img_mean[:, None, None]) / img_std_dev[:, None, None]

    return normalized


class MyDatasetMTL(Dataset):
    '''
    dir_path: path to data, having two folders named data and label respectively
    '''

    def __init__(self, dir_path, transform=None, in_chan=12):
        self.dir_path = dir_path
        self.transform = transform
        self.data_path = os.path.join(dir_path, "ms_images")
        self.data_lists = sorted(glob.glob(os.path.join(self.data_path, "*.tif")))
        self.label_path = os.path.join(dir_path, "label_7classes")
        self.label_lists = sorted(glob.glob(os.path.join(self.label_path, "*.tif")))
        self.edge_label_path = os.path.join(dir_path, "edge_label_7classes")
        self.edge_label_lists = sorted(glob.glob(os.path.join(self.edge_label_path, "*.tif")))
        self.in_chan = in_chan

    def __getitem__(self, index):
        img_path = self.data_lists[index]
        label_path = self.label_lists[index]
        edge_path = self.edge_label_lists[index]
        data_path = self.data_lists[index]

        img = read_img(img_path)
        edge = tiff.imread(edge_path)
        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        semantic_mask = label.copy()

        semantic_mask[semantic_mask == 213] = 0  # Pasture
        semantic_mask[semantic_mask == 97] = 1  # Primary natural vegetation
        semantic_mask[semantic_mask == 100] = 2  # Agriculture
        semantic_mask[semantic_mask == 176] = 3  # Mining
        semantic_mask[semantic_mask == 144] = 4  # Urban area
        semantic_mask[semantic_mask == 64] = 5  # Water body
        semantic_mask[semantic_mask == 63] = 6  # Other uses

        instance_mask = self.sem2ins(semantic_mask)
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

            img = self.transform(img)
            semantic_mask = self.transform(semantic_mask)
        else:
            # img = transforms.ToTensor()(img)
            img = torch.tensor(img, dtype=torch.float32)
            semantic_mask = torch.tensor(semantic_mask, dtype=torch.int32)

        # img = img.permute(1, 0, 2)
        img = img.to(device)
        semantic_mask = semantic_mask.to(device)
        instance_mask = torch.tensor(instance_mask, dtype=torch.float32).to(device)
        normal_edge_mask = torch.tensor(normal_edge_mask, dtype=torch.int32).to(device)
        # Returns
        return img, instance_mask, semantic_mask, normal_edge_mask, data_path

    def __len__(self):
        return len(self.data_lists)

    def sem2ins(self, label):
        seg_mask_g = label.copy()

        # seg_mask_g[seg_mask_g != 255] = 0
        # seg_mask_g[seg_mask_g == 255] = 1

        contours, hierarchy = cv2.findContours(seg_mask_g, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i in range(len(contours)):
            cnt = contours[i]
            seg_mask_g = cv2.drawContours(seg_mask_g, [cnt], 0, i + 1, -1)
        return seg_mask_g

    def generate_normal_edge_mask(self, edge):
        normal_edge_mask = edge.copy()

        normal_edge_mask[normal_edge_mask == 0] = 0  # Non edge
        normal_edge_mask[normal_edge_mask == 255] = 1  # Edge

        return normal_edge_mask

    # It will do same previous task, i.e., edge detection
    def generate_cluster_edge_mask(self, label):

        cluster_edge_mask = label.copy()

        cluster_edge_mask[cluster_edge_mask == 0] = 1
        cluster_edge_mask[cluster_edge_mask == 255] = 0
        cluster_edge_mask[cluster_edge_mask == 76] = 0
        cluster_edge_mask[cluster_edge_mask == 226] = 0
        cluster_edge_mask[cluster_edge_mask == 150] = 0
        cluster_edge_mask[cluster_edge_mask == 179] = 0
        cluster_edge_mask[cluster_edge_mask == 29] = 0

        return cluster_edge_mask


class MyDatasetSTL(Dataset):
    '''
    dir_path: path to data, having two folders named data and label respectively
    '''

    def __init__(self, dir_path, transform=None, in_chan=2):
        self.dir_path = dir_path
        self.transform = transform
        self.data_path = os.path.join(dir_path, "sar_images")
        self.data_lists = sorted(glob.glob(os.path.join(self.data_path, "*.tif")))
        self.label_path = os.path.join(dir_path, "semantic_label")
        self.label_lists = sorted(glob.glob(os.path.join(self.label_path, "*.tif")))
        self.in_chan = in_chan

    def __getitem__(self, index):
        # Getting item path
        img_path = self.data_lists[index]
        label_path = self.label_lists[index]
        data_path = self.data_lists[index]

        # Data Readings
        img = normalize(img_path)
        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        # Setting mask classes colours to interger values
        semantic_mask = label.copy()
        semantic_mask[semantic_mask == 213] = 0  # Pasture
        semantic_mask[semantic_mask == 97] = 1  # Primary natural vegetation
        semantic_mask[semantic_mask == 100] = 2  # Agriculture
        semantic_mask[semantic_mask == 176] = 3  # Mining
        semantic_mask[semantic_mask == 144] = 4  # Urban area
        semantic_mask[semantic_mask == 64] = 5  # Water body
        semantic_mask[semantic_mask == 63] = 6  # Other uses

        if self.transform is not None:
            seed = 666
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            if torch.backends.mps.is_available():
                torch.mps.manual_seed(seed)

            img = self.transform(img)
            semantic_mask = self.transform(semantic_mask)
        else:
            #img = transforms.ToTensor()(img)
            img = torch.tensor(img, dtype=torch.float32)
            semantic_mask = torch.tensor(semantic_mask, dtype=torch.int32)

        #img = img.permute(1, 0, 2)
        img = img.to(device)
        semantic_mask = semantic_mask.to(device)

        return img, semantic_mask, data_path

    def __len__(self):
        return len(self.data_lists)
