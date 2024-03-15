# Consulta de dados

import numpy as np
import skimage.io as skio
from glob import glob
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt


path = './Potsdam/label/top_potsdam_2_10_label_noBoundary_00.png'
#path = './histology/label/0TCGA-21-5786-01Z-00-DX1.png'

files = glob(path)
files.sort()

label = cv2.imread(path)
label = cv2.cvtColor(label, cv2.COLOR_RGBA2GRAY)
#label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

print(np.unique(label))

cluster_edge_mask = label.copy()

cluster_edge_mask[cluster_edge_mask == 0] = 1
cluster_edge_mask[cluster_edge_mask == 255] = 0
cluster_edge_mask[cluster_edge_mask == 76] = 0
cluster_edge_mask[cluster_edge_mask == 226] = 0
cluster_edge_mask[cluster_edge_mask == 150] = 0
cluster_edge_mask[cluster_edge_mask == 179] = 0
cluster_edge_mask[cluster_edge_mask == 29] = 0

#plt.imshow(cluster_edge_mask)
#plt.show()

""" 
semantic_mask[semantic_mask == 0] = 0
semantic_mask[semantic_mask == 255] = 1
semantic_mask[semantic_mask == 76] = 2
semantic_mask[semantic_mask == 226] = 3
semantic_mask[semantic_mask == 150] = 4
semantic_mask[semantic_mask == 179] = 5
semantic_mask[semantic_mask == 29] = 6

label_copy[label_copy == 0] = 0
label_copy[label_copy == 255] = 1
label_copy[label_copy == 76] = 2
label_copy[label_copy == 226] = 3
label_copy[label_copy == 150] = 4
label_copy[label_copy == 179] = 5
label_copy[label_copy == 29] = 6

plt.imshow(label_copy)
plt.show()

"""