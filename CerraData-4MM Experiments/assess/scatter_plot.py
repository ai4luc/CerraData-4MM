import os
import numpy as np
from glob import glob
import skimage.io as skio
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, confusion_matrix, accuracy_score


def image_pred(path_true_img, path_pred_img):
    # List
    pred = list()
    name_file = []

    # Path of the Images
    list_true = glob(path_true_img)
    list_true.sort()
    list_pred = glob(path_pred_img)
    list_pred.sort()

    for i in range(len(list_true)):
        # Data reading
        y_true = skio.imread(list_true[i])
        y_pred = skio.imread(list_pred[i])

        # F1-score
        f1score = f1_score(y_true.flatten(), y_pred.flatten(), average='macro')
        # Save the prediction
        pred.append(f1score)

    return pred



def plot_accuracy_comparison(sar_accuracy, optical_accuracy, sar_opt_accuracy):
    """
    This function receives SAR, Optical, and SAR+Opt accuracies and creates a scatter plot.

    Parameters:
    sar_accuracy (list of floats): Accuracy values for SAR
    optical_accuracy (list of floats): Accuracy values for Optical
    sar_opt_accuracy (list of floats): Accuracy values for SAR+Opt
    """

    # Check if all lists have the same length
    if not (len(sar_accuracy) == len(optical_accuracy) == len(sar_opt_accuracy)):
        print("All input lists must have the same length.")
        return

    # Create scatter plot
    plt.figure(figsize=(10, 6))

    # Plotting SAR vs Optical accuracy
    plt.scatter(sar_accuracy, optical_accuracy, color='#FFBF00', label='SAR vs MSI', s=60)

    # Plotting SAR vs SAR+Opt accuracy
    #plt.scatter(sar_accuracy, sar_opt_accuracy, color='#DE3163', label='SAR vs MSI+SAR', s=60)

    # Plotting Optical vs SAR+Opt accuracy
    #plt.scatter(optical_accuracy, sar_opt_accuracy, color='#008000', label='Opt vs Opt+SAR', s=60)

    # Adding labels and title
    plt.title("Scatter Plot of Accuracy Comparisons", fontsize=12)
    plt.xlabel("Accuracy", fontsize=10)
    plt.ylabel("Accuracy", fontsize=10)
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.savefig('/Users/mateus.miranda/INPE-CAP/PhD/Projetos/PrInt/Reports/assess/scatter_plot_transnuseg_SARvsMSI.jpeg',
                dpi=700, bbox_inches='tight', transparent=True)
    #plt.show()


# Paths
data_path = '/Users/mateus.miranda/INPE-CAP/PhD/Projetos/PhD_project/step1-2/assessment/visuais/'
print('Dados lidos')

# Prediction
sar_accuracy = image_pred(data_path + 'transnuseg/true/seg/*.tif', data_path + 'transnuseg/sar/seg/nonWN/*.tif')
optical_accuracy = image_pred(data_path + 'transnuseg/true/seg/*.tif', data_path + 'transnuseg/opt/seg/*.tif')
sar_opt_accuracy = image_pred(data_path + 'transnuseg/true/seg/*.tif', data_path + 'transnuseg/concat_data/nonWN/seg/*.tif')

# Call the function
plot_accuracy_comparison(sar_accuracy, optical_accuracy, sar_opt_accuracy)
