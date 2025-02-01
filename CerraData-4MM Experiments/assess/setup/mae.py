import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
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
from utils import * 

# Information of the Input image
img_WH = 128
img_channel = 14

# GPU information 
device = torch.device("mps")
print('Using ' + str(device) + ' device')

##### Architecture #####

# Transformers-based Autoencoder
class VisionTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(VisionTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x

# Encoder module: down-sampling
class VisionTransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_layers, num_heads, mlp_dim, patch_size=4, img_size=img_WH):
        super(VisionTransformerEncoder, self).__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.flatten_dim = patch_size * patch_size * input_dim
        self.linear_proj = nn.Linear(self.flatten_dim, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        self.layers = nn.ModuleList([
            VisionTransformerBlock(embed_dim, num_heads, mlp_dim)
            for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_patches, self.flatten_dim)
        x = self.linear_proj(x) + self.positional_encoding
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

# Decoder module: image up-sampling
class VisionTransformerDecoder(nn.Module):
    def __init__(self, output_dim, embed_dim, num_layers, num_heads, mlp_dim, patch_size=4, img_size=img_WH):
        super(VisionTransformerDecoder, self).__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.linear_proj = nn.Linear(embed_dim, patch_size * patch_size * output_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        self.layers = nn.ModuleList([
            VisionTransformerBlock(embed_dim, num_heads, mlp_dim)
            for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = x + self.positional_encoding
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.linear_proj(x)
        x = x.view(batch_size, self.img_size, self.img_size, -1).permute(0, 3, 1, 2)
        return x


# Connecting the modules
class MaskedAutoencoder(nn.Module):
    def __init__(self, input_dim=14, embed_dim=256, num_layers=6, num_heads=8, mlp_dim=512, patch_size=4, img_size=img_WH):
        super(MaskedAutoencoder, self).__init__()
        self.encoder = VisionTransformerEncoder(input_dim, embed_dim, num_layers, num_heads, mlp_dim, patch_size,
                                                img_size)
        self.decoder = VisionTransformerDecoder(input_dim, embed_dim, num_layers, num_heads, mlp_dim, patch_size,
                                                img_size)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Channels Masking Function
def mask_channels(x, p=0.5):
    """
    Selectively mask image channels.
        :param x: Input image tensor, shape [N, C, H, W]
        :param p: Probability of masking each channel.
        :return: Tensor with masked channels.
    """
    masked_x = x.clone()
    mask = torch.rand(x.size(1)) < p  # It generates a channels mask
    for i in range(x.size(1)):  # For each channel
        if mask[i]:
            masked_x[:, i, :, :] = 0  # It masks the channel
    return masked_x


# Save training history
def save_training_history(train_loss, val_loss, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()




