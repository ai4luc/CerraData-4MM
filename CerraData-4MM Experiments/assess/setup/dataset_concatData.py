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


def data_info(modality):
    # SAR statistical information
    if modality == 'sar':
        min = [0.16844511032105, 0.18629205226898335]
        max = [1877.8493041992167, 1303.7864481607917]
        mean = [104.87897585598166, 95.52668271493417]
        stddev = [79.8024668186095, 63.256644370816836]

        return max, min, mean, stddev

    # MSI statistical information
    elif modality == 'msi':
        min = [99.78856658935547, 332.65665627643466, 347.161809168756, 331.4168453961611,
               196.89053159952164, 240.9765984416008, 261.34731489419937, 342.50664601475,
               277.87501442432404, 246.40860325098038, 265.9057685136795, 226.23770987987518]
        max = [7349.042938232482, 8987.99301147458, 8906.377044677738, 9027.435272216775,
               9090.25390625, 8949.610290527282, 8955.640045166012, 9491.945373535062,
               9026.07144165042, 11857.606872558594, 11817.384948730469, 13970.691894531188]
        mean = [1331.2999603920011, 1422.618248839035, 1648.7418838236356, 1811.0396095371318,
                2243.6360604171587, 2862.469356914663, 3158.7246770243464, 3253.5804747400075,
                3464.1887187200564, 3463.5260019211623, 3635.662557047575, 2740.6395025025904]
        stddev = [436.04697715189127, 484.32797096427566, 549.125419913045, 741.2668466992163,
                  788.8006282648606, 860.9668486457188, 963.2983618801512, 1000.2677835011111,
                  1087.111000434025, 1062.9960118331512, 1373.6088616321088, 1125.5168224477407]

        return max, min, mean, stddev


def read_img(path):
    sar_img = gdal.Open(path)
    sar_i = sar_img.ReadAsArray()
    return sar_i


def normalizer(img_path, modality, norm_type):
    # Data loader
    data = read_img(img_path)
    normalized = 0

    # Data information
    max, min, mean, stddev = data_info(modality=modality)

    # Normalization between 0 and 1
    if norm_type == '0to1':
        for j in range(len(data)):
            normalized = (data[j] - min[j]) / (max[j] - min[j])
        return normalized

    # Normalization between -1 and 1
    elif norm_type == '1to1':
        for j in range(len(data)):
            normalized = ((np.log(data[j]) - np.log(mean[j])) / np.log(stddev[j]))
        return normalized
    else:
        print('Error: Normalization type if not available. Please, select either "0to1" or "1to1".')

        return 0


class MMDataset(Dataset):
    '''
    dir_path: path to data, having two folders named data and label respectively
    '''

    def __init__(self, dir_path, gpu, norm=False, transform=None):
        self.dir_path = dir_path
        self.transform = transform
        self.norm = norm
        self.device = gpu

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
        self.label_path = os.path.join(dir_path, "semantic_7c")
        self.label_lists = sorted(glob.glob(os.path.join(self.label_path, "*.tif")))
        # Edge mask
        self.edge_label_path = os.path.join(dir_path, "edge_7c")
        self.edge_label_lists = sorted(glob.glob(os.path.join(self.edge_label_path, "*.tif")))

    def __getitem__(self, index):
        # Getting item
        opt_img_path = self.opt_lists[index]
        sar_img_path = self.sar_lists[index]
        label_path = self.label_lists[index]
        edge_path = self.edge_label_lists[index]
        data_paths = self.data_lists[index]

        # Reading data
        if self.norm:
            opt_img = normalizer(img_path=opt_img_path, modality='msi', norm_type='0to1')
            sar_img = normalizer(img_path=sar_img_path, modality='sar', norm_type='0to1')
        else:
            opt_img = read_img(opt_img_path)
            sar_img = read_img(sar_img_path)

        semantic_mask = tiff.imread(label_path)
        edge_mask = tiff.imread(edge_path)

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
            edge_mask = self.transform(edge_mask)
        else:
            # Convert to the correct data type before converting to a tensor
            opt_img = torch.tensor(opt_img.astype(np.float32))
            sar_img = torch.tensor(sar_img.astype(np.float32))
            semantic_mask = torch.tensor(semantic_mask.astype(np.int32))
            edge_mask = torch.tensor(edge_mask.astype(np.float32))

        # Stack images along the channel dimension
        stacked_img = torch.cat((opt_img, sar_img), dim=0)

        stacked_img = stacked_img.to(self.device)
        semantic_mask = semantic_mask.to(self.device)
        edge_mask = edge_mask.to(self.device)

        return stacked_img, semantic_mask, edge_mask, data_paths

    def __len__(self):
        return len(self.opt_lists)


class MyDatasetMTL(Dataset):
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
        self.label_path = os.path.join(dir_path, "semantic_7c")
        self.label_lists = sorted(glob.glob(os.path.join(self.label_path, "*.tif")))
        # Edge mask
        self.edge_label_path = os.path.join(dir_path, "edge_7c")
        self.edge_label_lists = sorted(glob.glob(os.path.join(self.edge_label_path, "*.tif")))

    def __getitem__(self, index):
        opt_img_path = self.opt_lists[index]
        sar_img_path = self.sar_lists[index]
        label_path = self.label_lists[index]
        edge_path = self.edge_label_lists[index]
        data_paths = self.data_lists[index]

        #opt_img = normalize(opt_img_path, dtype='opt')
        #sar_img = normalize(sar_img_path, dtype='sar')
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

        stacked_img = stacked_img.to(device)
        semantic_mask = semantic_mask.to(device)
        normal_edge_mask = torch.tensor(normal_edge_mask, dtype=torch.int32).to(device)

        return stacked_img, semantic_mask, normal_edge_mask, data_paths

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


class MyDatasetSTL(Dataset):
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
        self.label_path = os.path.join(dir_path, "semantic_7c")
        self.label_lists = sorted(glob.glob(os.path.join(self.label_path, "*.tif")))

    def __getitem__(self, index):
        opt_img_path = self.opt_lists[index]
        sar_img_path = self.sar_lists[index]
        label_path = self.label_lists[index]
        data_paths = self.data_lists[index]

        #opt_img = normalize(opt_img_path, dtype='opt')
        #sar_img = normalize(sar_img_path, dtype='sar')
        opt_img = read_img(opt_img_path)
        sar_img = read_img(sar_img_path)
        semantic_mask = tiff.imread(label_path)

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
        else:
            opt_img = torch.tensor(opt_img, dtype=torch.float32)
            sar_img = torch.tensor(sar_img, dtype=torch.float32)
            semantic_mask = torch.tensor(semantic_mask, dtype=torch.int32)

        # Stack images along the channel dimension
        stacked_img = torch.cat((opt_img, sar_img), dim=0)

        stacked_img = stacked_img.to(device)
        semantic_mask = semantic_mask.to(device)

        return stacked_img, semantic_mask, data_paths

    def __len__(self):
        return len(self.opt_lists)

