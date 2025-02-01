# Data
import os
import numpy as np
from glob import glob
import skimage.io as skio
import cv2
import csv

# Metrics
from sklearn.metrics import f1_score, precision_score, confusion_matrix, accuracy_score, jaccard_score


# Set up

class7_list = {'pasture': 0, 'natural_vegetation': 1, 'agriculture': 2, 'mining': 3, 'built': 4, 'water': 5,
               'other_uses': 6}  # Cerradata-4MM 7 classes

class14_list = {'pasture': 0, 'primary_natural_vegetation': 1, 'secondary_natural_vegetation': 2,
                'water': 3, 'mining': 4, 'urban': 5, 'other_built': 6, 'forestry': 7, 'perennial_agri': 8,
                'semi_prennial_agri': 9, 'temp_1c_agri': 10, 'temp_1mais_agri': 11,
                'other_uses': 12, 'deforestation2022': 13}  # Cerradata-4MM 14 classes

lista_class_r = {value: key for key, value in zip(class14_list.keys(), class14_list.values())}


# Function to read CSV and calculate metrics
def evaluate_metrics_from_csv(csv_path):
    with open(csv_path, mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)

        # Extract confusion matrix from CSV
        confusion_matrix_data = np.array([list(map(int, row[1:])) for row in rows[1:]])

        # Calculate total F1-score
        y_true_flat = []
        y_pred_flat = []
        for i, row in enumerate(confusion_matrix_data):
            for j, count in enumerate(row):
                y_true_flat.extend([i] * count)
                y_pred_flat.extend([j] * count)

        total_f1_score = f1_score(y_true_flat, y_pred_flat, average='macro')

        # Calculate accuracy per class
        class_accuracies = {}
        for i in range(confusion_matrix_data.shape[0]):
            true_positive = confusion_matrix_data[i, i]
            total_samples = np.sum(confusion_matrix_data[i, :])
            accuracy = true_positive / total_samples if total_samples > 0 else 0
            class_accuracies[i] = accuracy * 100

    print(f"Total F1-Score: {total_f1_score:.4f}")
    print("Class-wise Accuracy (%):")
    for class_id, accuracy in class_accuracies.items():
        print(f"Class {class_id}: {accuracy:.2f}%")

    return total_f1_score, class_accuracies

csv_path = 'transnuseg_concat_w_l2_confusion_matrix.csv'
evaluate_metrics_from_csv(csv_path)
