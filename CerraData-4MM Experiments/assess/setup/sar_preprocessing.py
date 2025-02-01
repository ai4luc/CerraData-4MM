import numpy as np
import cv2
import tifffile as tiff
from glob import glob
import rasterio as rio

# from osgeo import gdal
""" 
def read_img(path):
    sar_img = gdal.Open(path)
    sar_i = sar_img.ReadAsArray()
    return sar_i

# Log 
def normalize(file_path):
    # Image reading 
    img = read_img(file_path)

    # Log transformation
    log_img = np.log1p(img)

    # Mean
    img_mean = np.array([5.0567, 4.4802])

    # Std Dev
    img_std_dev = np.array([0.4312, 0.4301])

    # Normalization
    normalized = (log_img - img_mean[:, np.newaxis, np.newaxis]) / img_std_dev[:, np.newaxis, np.newaxis]

    return normalized

"""


def read_img(path):
    # sar_img = gdal.Open(path)
    # sar_i = sar_img.ReadAsArray()
    sar_i = tiff.imread(path)
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
