import torch
import os
import numpy as np
from torch.utils.data import Dataset
import random
import glob
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from torchvision import transforms
import cv2
import tifffile as tiff


class MyDataset(Dataset):
    '''
    dir_path: path to data, having two folders named data and label respectively
    '''

    def __init__(self, dir_path, transform=None, num_classes=14):
        self.dir_path = dir_path
        self.transform = transform
        self.label_path = os.path.join(dir_path, "semantic_14c/")
        self.label_lists = sorted(glob.glob(os.path.join(self.label_path, "*.tif")))
        self.num_classes = num_classes

    def __getitem__(self, index):
        label_path = self.label_lists[index]
        label = tiff.imread(label_path)

        if self.transform is not None:
            seed = 666
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            # torch.cuda.manual_seed(seed)
            torch.mps.manual_seed(seed)
            label = self.transform(label)
        else:
            T = transforms.Compose([
                transforms.ToTensor()
            ])
            label = torch.tensor(label, dtype=torch.int32)

        return label

    def __len__(self):
        return len(self.label_lists)

DATA_PATH = "../datasets/cerradata4mm/"

# Initialize dataset and dataloader
dataset = MyDataset(dir_path=DATA_PATH, num_classes=14)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# Initialize a counter to count class occurrences
class_counter = Counter()

# Iterate over the dataset
for labels in dataloader:
    labels = labels.view(-1).numpy()  # Flatten the 4D tensor to 1D
    class_counter.update(labels.tolist())

# Compute the class distribution
total_pixels = sum(class_counter.values())
class_distribution = {cls: count / total_pixels for cls, count in class_counter.items()}

# Verifying that the sum of all class percentages is 100%
total_percentage = sum(class_distribution.values())

# Function to show absolute counts of each class
def show_absolute_class_counts(class_counter):
    print("Absolute Class Counts:")
    for cls, count in class_counter.items():
        print(f"Class {cls}: {count}")

# Print the class counts and formatted class distributions
print(f"Class Counts: {class_counter}")
print("Class Distribution (formatted):")
for cls, dist in class_distribution.items():
    print(f"Class {cls}: {dist:.4f}")

print(f"Total Percentage: {total_percentage}")

# Show absolute counts of each class
show_absolute_class_counts(class_counter)

# Classes frequency
class_frequencies = {cls: count / total_pixels for cls, count in class_counter.items()}

# Assign weights (inverse to the classes frequency)
class_weights = {cls: 1.0 / freq for cls, freq in class_frequencies.items()}

# Weights normalization
max_weight = max(class_weights.values())
class_weights = {cls: weight / max_weight for cls, weight in class_weights.items()}

# To tensor
weights_tensor = torch.tensor([class_weights[i] for i in range(len(class_weights))], dtype=torch.float)

print("Class Weights:", {cls: round(weight, 4) for cls, weight in class_weights.items()})
print("Weights Tensor:", weights_tensor)
