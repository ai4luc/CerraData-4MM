import cv2
import os
from glob import glob
import numpy as np
import rasterio as rio
from rasterio.merge import merge

print('Oi')

# Mask paths
label_path = '../datasets/cerradata4mm_exp/train/semantic_7mas/*.tif'
output_path = '../datasets/cerradata4mm_exp/train/semantic_7c/'

list_files = glob(label_path)
list_files.sort()

# EDGE infor
color_to_edgclass = {
    (255, 255, 255): 1,   # edge
    (0, 0, 0): 0    # background
}

# RGB and labels map
color_to_7class = {
    (206, 239, 98): 0,   # Pasture
    (22, 152, 13): 1,    # Arboreal
    (230, 32, 108): 2,   # Agriculture
    (176, 176, 176): 3,  # Mining
    (223, 124, 38): 4,   # Urban area
    (19, 50, 255): 5,    # Water body
    (117, 10, 194): 6    # Other uses
}

color_to_14class = {
    (206, 239, 98): 0,   # Pasture
    (22, 152, 13): 1,    # Primary natural vegetation
    (31, 212, 18): 2,    # Secondary natural vegetation
    (19, 50, 255): 3,    # Water
    (176, 176, 176): 4,  # Mining
    (223, 124, 38): 5,   # Urban area
    (250, 128, 114): 6,  # Other Built area
    (85, 107, 47): 7,    # Forestry
    (230, 32, 108): 8,   # Perennial Agriculture
    (139, 105, 20): 9,   # Semi-perennial agriculture
    (255, 215, 0): 10,   # Temporary agriculture of 1 cycle
    (255, 255, 0): 11,   # Temporary agriculture of +1 cycle
    (117, 10, 194): 12,  # Other uses
    (205, 0, 0): 13      # Deforestation 2022
}

# Convert the dictionary to an array
colors = np.array(list(color_to_7class.keys()))
classes = np.array(list(color_to_7class.values()))
print(f'{list_files}')
for filename in list_files:
    
    # File information
    file_name = os.path.basename(filename)

    # Get geo information
    geo_label = rio.open(filename)
    mosaic, out_trans = merge([geo_label])
    out_meta = geo_label.meta.copy()
    #crs = rio.crs.CRS({'init':'epsg:4326'})
    crs = geo_label.crs

    # Image loading
    label = cv2.imread(filename)
    if label is None:
        print(f"Failed to load image {filename}")
        continue

    # Convert BGR to RGB
    label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
    
    # Create a temporary image
    semantic_mask = np.zeros(label.shape[:2], dtype=np.int32)

    # Assigning labels
    for color, class_id in zip(colors, classes):
        mask = np.all(label == color, axis=-1)
        semantic_mask[mask] = class_id

    print(np.unique(semantic_mask))
    print(np.shape(semantic_mask))

    # Label saving
    save_path = os.path.join(output_path, file_name)
    print(save_path)
    with rio.open(save_path, mode='w', driver='GTiff', height=semantic_mask.shape[0], width=semantic_mask.shape[1], count=1, dtype='int32', crs=crs, transform=out_trans) as dest:
        dest.write(semantic_mask, 1)

