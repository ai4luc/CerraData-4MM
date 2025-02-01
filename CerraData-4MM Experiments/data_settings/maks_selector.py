import os
import glob
import numpy as np
import cv2
import shutil

import os
import glob
import numpy as np
import cv2
import shutil

# Caminho das máscaras
mask_path = '../datasets/cerradata4c_draft/label_10classes/*.tif'
rejected_mask_path = '../datasets/cerradata4c_draft/rejected_labels/'

# Lista dos arquivos de imagens
list_file = glob.glob(mask_path)
# Ordenando os arquivos
list_file.sort()

# Loop para processar cada imagem
for im, sample in enumerate(list_file, start=1):
    print('Img: ', im)

    label = cv2.imread(sample)
    label = cv2.cvtColor(label, cv2.COLOR_RGBA2GRAY)

    semantic_mask = label.copy()
    pxclass = np.unique(semantic_mask)

    # Verificar se existe pixel com valor 0
    if 0 in pxclass:
        # Obter o nome do arquivo e criar o caminho de destino
        filename = os.path.basename(sample)
        dest_path = os.path.join(rejected_mask_path, filename)

        # Mover a imagem para a pasta de rejeitados
        shutil.move(sample, dest_path)
        print(f'Movido para {dest_path}')
    else:
        print(f'Nenhum pixel 0 encontrado na imagem {sample}')
