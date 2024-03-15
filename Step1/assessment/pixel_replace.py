
import matplotlib.pyplot as plt
import os
import numpy as np
from glob import glob
import skimage.io as skio
import cv2
from sklearn.metrics import f1_score


# Data loading
def load_image(path):
    # List
    patch = list()
    name_file = []
    # Path of the Images
    list_file = glob(path)
    # Sorting name file
    list_file.sort()

    # Loop
    for sample in list_file:
        # name of file
        name_file.append(os.path.basename(sample))

        # Read Image
        raster = skio.imread(sample)
        patch.append(raster)

    return patch, name_file

# Paths
path_data = '../models/deeplabv3plus/150epo_output/ce_dice/'
path_true = '../dataset/Potsdam/Potsdam_split_data2/test/seg/'

path_save_data = '/Users/mateus.miranda/INPE-CAP/PhD/Document/Proposta/Projeto/exp1/assessment/visuais/deeplab/'

y_true, y_true_name = load_image(path_true + '*.tif')
y_pred, y_pred_name = load_image(path_data + '*.tif')
print('Dados lidos')


# define color map Seg

color_map = {0: np.array([255, 255, 255]), # impervious_surface
             1: np.array([0, 0, 255]), # building
             2: np.array([0, 255, 255]), # low_vegetation
             3: np.array([0, 255, 0]), # tree
             4: np.array([255, 255, 0]), # car
             5: np.array([255, 0, 0])} # background
"""
# define color map Edge
color_map = {0: np.array([255, 255, 255]), # impervious_surface
             1: np.array([0, 0, 0])} # background
"""

for k in range(len(y_true)):
    # Metric
    f1score = f1_score(y_true[k].flatten(), y_pred[k].flatten(), average='weighted')

    if f1score >= 0.85:
        # Saving
        img = np.array(y_pred[k]).reshape(512, 512)
        data_3d = np.zeros(shape=(img.shape[0], img.shape[1], 3), dtype=int)
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                data_3d[i][j] = color_map[img[i][j]]

        # Save the image
        data_3d = np.array(data_3d)
        #print(data_3d.shape)
        #print(data_3d.dtype)

        fig, ax = plt.subplots(1, 1)
        ax.imshow(data_3d)
        #plt.savefig(path_save_data + 'Unet_seg_better_'+str(k))

    elif f1score <=0.5:
        # Saving
        img = np.array(y_pred[k]).reshape(512, 512)
        data_3d = np.zeros(shape=(img.shape[0], img.shape[1], 3), dtype=int)
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                data_3d[i][j] = color_map[img[i][j]]

        # Save the image
        data_3d = np.array(data_3d)
        print(data_3d.shape)
        print(data_3d.dtype)

        fig, ax = plt.subplots(1, 1)
        ax.imshow(data_3d)
        plt.savefig(path_save_data + 'DeepLab_seg_worst_' + str(k))
    else:
        pass

""" 
    # Saving
    img = np.array(img).reshape(512,512)
    data_3d = np.zeros(shape=(img.shape[0], img.shape[1], 3), dtype=int)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            data_3d[i][j] = color_map[img[i][j]]

    # Save the image
    data_3d = np.array(data_3d)
    print(data_3d.shape)
    print(data_3d.dtype)

    # display the plot
    fig, ax = plt.subplots(1,1)
    ax.imshow(data_3d)
    plt.savefig(path_save_data+data_name[k])
"""



