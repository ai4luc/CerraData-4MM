"""
Software Open Source
©Copyright Mateus de Souza Miranda, 2023
mateus.miranda@inpe.br; mateusmirandaa2@hotmail.com
National Institute for Space Research (INPE)
São José dos Campos, São Paulo, Brazil

Version 4:  January, 2023
"""

# -------- Packs --------
# Data
import os
import numpy as np
from glob import glob
import skimage.io as skio
import cv2
import csv

# Metrics
from sklearn.metrics import f1_score, precision_score, confusion_matrix, accuracy_score, jaccard_score

# Graphs
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# -------- Pixel-based assessment --------
# Set up

class7_list = {'pasture': 0, 'natural_vegetation': 1, 'agriculture': 2, 'mining': 3, 'built': 4, 'water': 5,
               'other_uses': 6}  # Cerradata-4MM 7 classes

class14_list = {'pasture': 0, 'primary_natural_vegetation': 1, 'secondary_natural_vegetation': 2,
                'water': 3, 'mining': 4, 'urban': 5, 'other_built': 6, 'forestry': 7, 'perennial_agri': 8,
                'semi_prennial_agri': 9, 'temp_1c_agri': 10, 'temp_1mais_agri': 11,
                'other_uses': 12, 'deforestation2022': 13}  # Cerradata-4MM 14 classes

lista_class_r = {value: key for key, value in zip(class14_list.keys(), class14_list.values())}


# Metrics

def dice_coefficient(predicted_mask, ground_truth_mask):
    """
  Função para calcular o Coeficiente de Dice entre duas máscaras.

  Argumentos:
    predicted_mask: Máscara prevista pelo algoritmo de segmentação.
    ground_truth_mask: Máscara real do objeto de interesse.

  Retorno:
    Coeficiente de Dice entre as duas máscaras.
  """

    # Calcular a interseção entre as duas máscaras.
    intersection = np.sum(predicted_mask * ground_truth_mask)

    # Calcular a soma dos elementos de ambas as máscaras.
    sum_masks = np.sum(predicted_mask) + np.sum(ground_truth_mask)

    # Evitar divisão por zero.
    if sum_masks == 0:
        return 0

    # Calcular o Coeficiente de Dice.
    dice = (2 * intersection) / sum_masks

    return dice


def compute_iou(label, pred):
    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels)
    iou = {}

    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I = float((label_i & pred_i).sum())
        U = float((label_i | pred_i).sum())
        iou[lista_class_r[val]] = (I / U)

    return iou


def compute_mean_iou(pred, label):
    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels)
    iou = {}

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))
        iou[lista_class_r[val]] = (I[index] / U[index])  # adicionei - Marcos

    mean_iou = np.mean(I / U)
    return mean_iou, iou


def compute_mean_iou_weighted(label, pred):
    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels)
    iou = {}
    mIoU_weighted = 0.0

    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I = float((label_i & pred_i).sum())
        U = float((label_i | pred_i).sum())
        iou[lista_class_r[val]] = (I / U)

        mIoU_weighted += (iou[lista_class_r[val]] * (label_i).sum())

    mIoU_weighted = mIoU_weighted / label.ravel().shape[0]

    return iou, mIoU_weighted


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
        #raster = np.array(raster).reshape(512, 512)
        patch.append(raster)

    return patch, name_file


# Paths
#path2PlotSave = 'report/confusion_matrix/unet_W7C_concat.jpeg'
path_true = '../datasets/cerradata4mm_exp/test/semantic_7c/*.tif'
path_pred = '../models/concat/transnuseg/pred_7/w/seg/*.tif'

y_true, name_ytrue = load_image(path_true)
y_pred, name_ypred = load_image(path_pred)

y_pred = np.array(y_pred)
y_true = np.array(y_true)

val, qtd = np.unique(y_true, return_counts=True)

# Save confusion matrix to CSV file
output_csv_path = 'transnuseg_concat_w_l1_confusion_matrix.csv'

"""
### Metrics
# 1. IoU and mIoU
iou = compute_iou(y_true, y_pred)
#print(f"IoU:{iou}")

meanIoU, iou2 = compute_mean_iou(y_pred, y_true)
print(f"mean IoU:{meanIoU}")
#print(f"IoU:{iou2}")

# 2. F1-score
f1score = f1_score(y_true.flatten(), y_pred.flatten(), average='macro')
print('F1-score Macro:', float(f1score))

f1score_weighted = f1_score(y_true.flatten(), y_pred.flatten(), average='weighted')
print('F1-score Weighted:', float(f1score_weighted))

# 3. Accuracy
#acc = accuracy_score(y_true.flatten(), y_pred.flatten())
#print('Accuracy:', acc)
"""
# 5. Confusion Matrix
confusionMatrix = confusion_matrix(y_true.flatten(), y_pred.flatten())




with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write header row (class names or indices)
    header = ["Class"] + list(range(confusionMatrix.shape[1]))
    writer.writerow(header)

    # Write each row of the confusion matrix
    for i, row in enumerate(confusionMatrix):
        writer.writerow([i] + row.tolist())

print(f"Confusion matrix saved to {output_csv_path}")

""" 
confusionMatrixPercent = confusionMatrix.astype('float') / confusionMatrix.sum(axis=1)[:, np.newaxis] * 100

# Plot
plt.figure(figsize=(8, 6))
plt.imshow(confusionMatrixPercent, interpolation='nearest', cmap=plt.cm.Greys)
#plt.title('U-Net on SAR nonwn')
plt.colorbar()
classes7 = [0, 1, 2, 3, 4, 5, 6]
classes14 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

tick_marks = np.arange(len(classes7))
plt.xticks(tick_marks, classes7, rotation=45)
plt.yticks(tick_marks, classes7)
plt.ylabel('True')
plt.xlabel('Predicted')

for i in range(confusionMatrixPercent.shape[0]):
    for j in range(confusionMatrixPercent.shape[1]):
        plt.text(j, i, format(confusionMatrixPercent[i, j], '.2f') + '%',
                 ha="center", va="center",
                 color="white" if confusionMatrixPercent[i, j] > 50 else "black")

plt.tight_layout()
plt.savefig(path2PlotSave, dpi=700, transparent=True)

# Confusion matrix on absolut values

# 5. Confusion Matrix
confusionMatrix = confusion_matrix(y_true.flatten(), y_pred.flatten())

# Plot
plt.figure(figsize=(18, 16))
plt.imshow(confusionMatrix, interpolation='nearest', cmap=plt.cm.Greys)
#plt.title('U-Net on SAR nonwn')
plt.colorbar()
classes7 = [0, 1, 2, 3, 4, 5, 6]
classes14 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

tick_marks = np.arange(len(classes14))
plt.xticks(tick_marks, classes14, rotation=45)
plt.yticks(tick_marks, classes14)
plt.ylabel('True')
plt.xlabel('Predicted')

for i in range(confusionMatrix.shape[0]):
    for j in range(confusionMatrix.shape[1]):
        plt.text(j, i, str(confusionMatrix[i, j]),
                 ha="center", va="center",
                 color="white" if confusionMatrix[i, j] > confusionMatrix.max() / 2 else "black")

plt.tight_layout()
plt.savefig(path2PlotSave, dpi=700, transparent=True)

"""

