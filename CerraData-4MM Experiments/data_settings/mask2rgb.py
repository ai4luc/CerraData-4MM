import cv2
import os
from glob import glob
import numpy as np
import rasterio as rio
from rasterio.merge import merge

# Paths
input_mask_path = '../models/concat/unet/pred_7/nonw/parrot_beak_unet_29189.tif'
output_rgb_path = '../models/concat/unet/pred_7/rgb_nw/'

os.makedirs(output_rgb_path, exist_ok=True)

# RGB mapping (inverso do dicionário color_to_7class)
class7_to_color = {
    0: (206, 239, 98),   # Pastagem
    1: (22, 152, 13),    # Arbóreo
    2: (230, 32, 108),   # Agricultura
    3: (176, 176, 176),  # Mineração
    4: (223, 124, 38),   # Área urbana
    5: (19, 50, 255),    # Corpo d'água
    6: (117, 10, 194)    # Outros usos
}

class14_to_color = {
     0: (206, 239, 98),   # Pasture
     1: (22, 152, 13),    # Primary natural vegetation
     2: (31, 212, 18),    # Secondary natural vegetation
     3: (19, 50, 255),    # Water
     4: (176, 176, 176),  # Mining
     5: (223, 124, 38),   # Urban area
     6: (250, 128, 114),  # Other Built area
     7: (85, 107, 47),    # Forestry
     8: (230, 32, 108),   # Perennial Agriculture
     9: (139, 105, 20),   # Semi-perennial agriculture
     10: (255, 215, 0),   # Temporary agriculture of 1 cycle
     11: (255, 255, 0),   # Temporary agriculture of +1 cycle
     12: (117, 10, 194),  # Other uses
     13: (205, 0, 0)      # Deforestation 2022
}

# Convertendo para arrays para eficiência
classes = np.array(list(class7_to_color.keys()))
colors = np.array(list(class7_to_color.values()))

# Obter arquivos de máscara
list_files = glob(input_mask_path)
list_files.sort()

for filename in list_files:
    # Obter informações do arquivo
    file_name = os.path.basename(filename)

    # Ler máscara como array
    with rio.open(filename) as src:
        mask = src.read(1)  # Assumindo que a máscara tem um único canal
        crs = src.crs
        transform = src.transform

    # Inicializar imagem RGB
    rgb_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Mapear classes para cores
    for class_id, color in zip(classes, colors):
        rgb_image[mask == class_id] = color

    # Salvar a imagem RGB
    save_path = os.path.join(output_rgb_path, file_name)
    with rio.open(save_path, mode='w', driver='GTiff',
                  height=rgb_image.shape[0], width=rgb_image.shape[1],
                  count=3, dtype='uint8', crs=crs, transform=transform) as dest:
        dest.write(rgb_image[:, :, 0], 1)  # Canal R
        dest.write(rgb_image[:, :, 1], 2)  # Canal G
        dest.write(rgb_image[:, :, 2], 3)  # Canal B

    print(f"Imagem salva em: {save_path}")
