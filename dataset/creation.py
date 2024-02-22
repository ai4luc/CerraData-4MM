from torch.utils.data import DataLoader
from dataset import MyDataset
import torch
import numpy as np

POTSDAM_DATA_PATH = '../dataset/Potsdam/Potsdam_dataset2/'

data_path = POTSDAM_DATA_PATH
total_data = MyDataset(dir_path=data_path)

datal = torch.utils.data.DataLoader(total_data, batch_size=1, shuffle=False)

print('Reading data')
for mask in datal:
    print(np.unique(mask))


