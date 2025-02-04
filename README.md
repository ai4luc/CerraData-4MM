![image](visual_setup/Head.png)

# CerraData-4MM - A Multimodal Dataset on Cerrado for Land Use and Land Cover Classification

CerraData-4MM is a multimodal dataset focusing on the Bico do Papagaio ecoregion within the Cerrado biome. This dataset is designed to support regional studies and applications aimed at the conservation and management of this unique and biodiverse biome, which lies in a transitional area with the Amazon forest. The dataset includes two modalities of data: **multispectral patches** (12 channels) and **Synthetic Aperture Radar (SAR) patches** (2 channels). Each modality contains **30,291 image patches** of size **128x128 pixels** with a spatial resolution of **10 meters**. The dataset is enriched by its hierarchical organization into two levels of classes, making it a valuable resource for land use and land cover classification tasks.

---

## About the Dataset

CerraData-4MM integrates **SAR** and **multispectral imagery** from the **Sentinel-1 (S1)** and **Sentinel-2 (S2)** satellites, respectively. The images were collected in **2022** and are aligned with reference data based on a land use and land cover map produced by the **TerraClass Cerrado program** in 2022. The dataset is particularly challenging at the second class level, especially regarding vegetation regeneration stages and types of agriculture, as the categories exhibit similar patterns.

### Key Features:
- **Two Modalities**:
  - Multispectral patches (12 channels)
  - SAR patches (2 channels)
- **Image Patches**: 30,291 patches per modality
- **Patch Size**: 128x128 pixels
- **Spatial Resolution**: 10 meters
- **Hierarchical Classes**:
  - **Level 1 (L1)**: 7 classes
  - **Level 2 (L2)**: 14 classes

### Class Hierarchy

The dataset contains two hierarchical levels of classes: **Level 1 (L1)** and **Level 2 (L2)**. Below is the complete list of classes, their corresponding IDs, and their RGB colors code.

| **L1 Class**      | **L1 ID** | **L2 Class**                                | **L2 ID** | **RGB**   |
|--------------------|-----------|---------------------------------------------|-----------|----------------|
| Pasture           | 0         | Pasture (Pa)                                | 0         | 206, 239, 98   |
| Forest            | 1         | Primary Natural Vegetation (V1)             | 1         | 22, 152, 13    |
| Agriculture       | 2         | Secondary Natural Vegetation (V2)           | 2         | 31, 212, 18    |
| Mining            | 3         | Mining (Mg)                                 | 4         | 176, 176, 176  |
| Building          | 4         | Urban area (UA)                             | 5         | 223, 124, 38   |
| Water body        | 5         | Water body (Wt)                             | 3         | 19, 50, 255    |
| Other Uses        | 6         | Other Built area (OB)                       | 6         | 250, 128, 114  |
|                   |           | Forestry (Ft)                               | 7         | 85, 107, 47    |
|                   |           | Perennial Agriculture (PR)                  | 8         | 230, 32, 108   |
|                   |           | Semi-perennial Agriculture (SP)             | 9         | 139, 105, 20   |
|                   |           | Temporary agriculture of 1 cycle (T1)       | 10        | 255, 215, 0    |
|                   |           | Temporary agriculture of 1+ cycle (T1+)     | 11        | 255, 255, 0    |
|                   |           | Other Uses (OU)                             | 12        | 117, 10, 194   |
|                   |           | Deforestation 2022 (Df)                     | 13        | 205, 0, 0      |



CerraData-4MM comprises sets of SAR and MSI data, followed by semantic maps for L1 and L2, respectively. The two class levels create a diverse and challenging dataset that covers categories of regeneration level, deforestation increment, as well as different types of agriculture. Each set contains 30,322 patches with a spatial resolution of 10 meters. This dataset provides a rich diversity of classes representative of the \textit{Bico do Papagaio} ecoregion. As illustrated in Figure below, L2 introduces five additional subcategories of agricultural types, one additional category for built areas, and two distinct vegetation generation classes. 

![image](visual_setup/dataset_git.png)

---

## Tutorial

### Data Downloading
You can download the dataset using the `kagglehub` Python package. Below is an example of how to download the latest version of the dataset:

```python
import kagglehub

# Download the latest version
path = kagglehub.dataset_download("cerranet/cerradata-4mm")

print("Path to dataset files:", path)

```


### Data Loading 
First, you can split the data into sub-directories for training, testing, and validating phases. Hence, consider the following code example, in which we are splitting the data into train and testing sub-directories. 

```python
import splitfolders

# Split data into Train and testing

input_folder = '../../cerradata-4mm/'
output_folder = '../../cerradata4mm_exp/'

# Ratio of split are in order of train/val/test.
splitfolders.ratio(input_folder, output_folder, seed=42, ratio=(.9, .0, .1))

```

Now, let's load the data! Find the function on the `dataset-loader` within the `CerraData-4MM Experiments/util/` directory, and then import the `MMDataset()` to read both modalities or `SARDataset()` `MSIDataset()` to read one of the modalities. In order to exemplify how to load the data from the `train` subset, let's consider `MMDataset()` class. 


```python
import torch
import os
import sys
from torch.utils.data import DataLoader, random_split
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'util')))
from dataset_loader import MMDataset

# GPU
## On NVIDIA architecture
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

## On Apple M chip architecture
#device = torch.device("mps")

# Path
DATA_PATH = '../../cerradata4mm_exp/train/'

# Data loading
total_data = MMDataset(dir_path=DATA_PATH, gpu=device, norm='none') # norm: "none", "0to1", "1to1"

# Split into Training and Validation phase
train_set_size = int(len(total_data) * 0.8)
val_set_size = len(total_data) - train_set_size
train_set, val_set = random_split(total_data, [train_set_size, val_set_size], generator=torch.Generator().manual_seed(random_seed))

trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

dataloaders = {"train": trainloader, "val": valloader}
dataset_sizes = {"train": len(trainloader), "val": len(valloader)}

```


### Training a model
Let's play a bit of training U-Net on CerraData-4MM. In the following summarized example, the model is training on the first hierarchical level, i.e., 7 classes, considering both modalities, hence 14 channels. 

```python
import torch
from torch import optim
import segmentation_models_pytorch as smp
import copy


# Define the U-Net model architecture:
# - encoder_name='resnet50': Uses ResNet50 as the encoder backbone
# - encoder_weights=None: No pre-trained weights for the encoder
# - in_channels=14: Input has 14 channels (e.g., multispectral data)
# - classes=7: Output has 7 classes for segmentation
# - activation='softmax': Applies softmax on outputs (potential issue)
model = smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=14, classes=7, activation='softmax')

# Initialize a copy of the model's weights to track the best-performing version
best_model_wts = copy.deepcopy(model.state_dict())


# Define the loss function with class weights for imbalanced data
# - weight=weight_tensor: Tensor assigning weights to each class (must be predefined, thus check it out in `CerraData-4MM Experiments/models` directory)
criterion = torch.nn.CrossEntropyLoss(weight=None)


# Define the Adam optimizer to update model parameters
# - lr=0.001: Learning rate (default for Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Track the best validation loss and its corresponding epoch
best_loss = float('inf')  # Initialize with a high value
best_epoch = 0  # Epoch where the best loss occurred


# Move the model to the GPU (if available)
model.to(device)


# Training loop for 100 epochs (or until early stopping)
for epoch in range(100):
    print(f'====== Epoch {epoch} ======')
    
    # Early stopping: Break if no improvement in 50 epochs
    if epoch > best_epoch + 50:
        break
    
    # Iterate over training and validation phases
    for phase in ['train', 'val']:
        running_loss = 0.0  # Cumulative loss for the epoch
        
        # Set model to training/evaluation mode
        if phase == 'train':
            model.train()  # Enables dropout/batch normalization
        else:
            model.eval()   # Disables dropout/batch normalization


        # Process each batch in the DataLoader
        for img, semantic_seg_mask, _ in dataloaders[phase]:
            # Convert input images to float and move to GPU
            img = img.float().to(device)
            
            # Move segmentation masks to GPU
            semantic_seg_mask = semantic_seg_mask.to(device)


            # Reset optimizer gradients
            optimizer.zero_grad()


            # Enable gradients only during training
            with torch.set_grad_enabled(phase == 'train'):
                # Forward pass: model predictions
                output = model(img)
                
                # Compute loss between predictions and ground truth
                # .long() converts masks to integer type
                loss = criterion(output, semantic_seg_mask.long())


                # Backpropagation and optimization only in training phase
                if phase == 'train':
                    loss.backward()    # Compute gradients
                    optimizer.step()   # Update model weights


            # Update running loss (weighted by batch size)
            running_loss += loss.item() * img.size(0)


        # Calculate epoch loss (average over all samples)
        epoch_loss = running_loss / dataset_sizes[phase]
        print(f'{phase} Loss: {epoch_loss:.4f}')
        
        # Log loss to file
        logging.info(f'{phase} Loss: {epoch_loss:.4f}')


        # Append loss to training/validation history
        if phase == 'train':
            train_loss.append(epoch_loss)
        else:
            val_loss.append(epoch_loss)


        # Update best model if validation loss improves
        if phase == 'val' and epoch_loss < best_loss:
            best_loss = epoch_loss       # Update best loss
            best_epoch = epoch           # Update best epoch
            best_model_wts = copy.deepcopy(model.state_dict())  # Save weights
            
            # Save best model to disk
            torch.save(best_model_wts, f'/home/.../unet_7cMSISAR_NW_model_epoch_{epoch}.pt')
            print(f'Best model saved at epoch {epoch} with loss {best_loss:.4f}')

    
    # Clear GPU memory cache after each epoch
    torch.cuda.empty_cache()

    # Load best model weights for evaluation
    model.load_state_dict(best_model_wts)
    model.eval()


    # Calculate Dice score on validation set
    dice_acc_val = 0
    dice_loss_val = DiceLoss(num_classes)  # Custom Dice loss

    with torch.no_grad():
        print('Validation...')
        for img, semantic_seg_mask, _ in valloader:
            img = img.float().to(device)
            semantic_seg_mask = semantic_seg_mask.to(device)
            output1 = model(img)
            
            # Print unique predicted and ground truth values (for debugging)
            print('Pred', np.unique(torch.argmax(output1, dim=1).cpu().numpy().ravel()))
            print('True', np.unique(semantic_seg_mask.cpu().numpy().ravel()))
            
            # Compute Dice loss (softmax=True may conflict with model's softmax)
            d_l = dice_loss_val(output1, semantic_seg_mask.float(), softmax=True)
            dice_acc_val += 1 - d_l.item()
```


### Run a model
To download the trained models you have to: 
1 - Get the Google Drive file ID from the shareable link, e.g., https://drive.google.com/file/d/XYZ123/view → ID is XYZ123 (See in CerraData-4MM Experiments/models Section). 
2 - Use the direct download URL with confirm=1 to bypass the virus scan warning page.
3 - Use requests to download and save the file.

```python
import requests

def download_public_zip(file_id, output_path):
    # Direct download URL template
    url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=1"

    # Send a GET request to download the file
    session = requests.Session()
    response = session.get(url, stream=True)

    # Save the content to a ZIP file
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
            if chunk:
                f.write(chunk)

# Obtaining the data
file_id = "YOUR_FILE_ID"  # Replace with the Google Drive file ID
output_path = "downloaded_file.zip"
download_public_zip(file_id, output_path)
```

Afterwards, you are able to load them. For example, 

```python
import torch
import os
import sys
from torch.utils.data import DataLoader, random_split
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'util')))
from dataset_loader import MMDataset
import segmentation_models_pytorch as smp
import rasterio
from rasterio.merge import merge

# Define the GPU device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Define the path of the test subset
DATA_PATH = '../../cerradata4mm_exp/test/'

# Data loading
test_msisar = MMDataset(dir_path=DATA_PATH, gpu=device, norm='none')  #Norm: "none" "1to1" "0to1"
test_msisar_load = torch.utils.data.DataLoader(test_msisar, batch_size=1, shuffle=False)

# Model loading
path_trained_model = 'CerraData-4MM Experiments/models/concat/unet/saved_model_7c/unet_7cMSISAR_2nonW_model_epoch_29.pt'

# Path to save the predictions 
output_path = 'CerraData-4MM Experiments/models/concat/unet/pred_7c/'

#  Load best model weights for testing
model.load_state_dict(torch.load(path_trained_model, map_location=device))
model.eval()

# Move the model to the GPU (if available)
model.to(device)

with torch.no_grad():
    # Data loading to GPU
    inputs = data.to(device)

    # Prediction
    y_pred = model(inputs)

    # GPU to CPU
    y_predmax = y_pred.argmax(1).cpu().numpy()

# To array
unet_output = np.array(y_predmax, dtype='int32').reshape(128, 128)

# Saving
save_path = output_path + 'parrot_beak_unet_nw_' + str(patch_ids) + '.tif'
with rasterio.open(save_path,
                   mode="w", driver='GTiff',
                   height=128, width=128,
                   count=1, dtype='int32',
                   crs=crs, transform=out_trans
                   ) as dest:
    dest.write(unet_output, 1)

```
Access the directory `CerraData-4MM Experiments/models` to access complete information on how to train your model in CerraData-4MM, as well as access those models that have already been trained.

---
## Standard Benchmarks
We selected the U-Net and TransNuSeg models to experiment with the dataset. The models' performance scores are presented below. The figure presents an overview of the overall results for each baseline model across the diverse scenarios. It incorporates scores that indicate the utilization (W) and non-utilization (NW) of class weights for dataset balancing. For instance, the notation “MSI-NW-L1” denotes the modality set without class weights at the initial hierarchical level of classes. TransNuSeg outperforms U-Net in most scenarios, particularly at the L1 hierarchical level, while the ViT-based model demonstrates better multimodal learning.  Combining MSI and SAR modalities improves performance compared to using either individually, but scores drop significantly for more complex L2 classes. For instance, in the Concat-NW-L1 scenario, the ViT-based model achieves an F1-score of 57.60%, whereas for the “MSI-NW-L1” and “SAR-NW-L1” scenarios, the corresponding scores are 57.45% and 38.94%, respectively. In the “MSI-NW-L2” scenario, TransNuSeg achieves an F1-score of 57.45% and an mIoU of 48.94% in L1, but these metrics decline to 53.17% and 41.90%, respectively.
U-Net exhibits even more unsatisfactory performance; in the “Concat-NW-L2” scenario, its F1-score and mIoU decrease to 18.16% and 15.84%, respectively.

![image](visual_setup/models_performance.png)

Read the full paper at <>.

---
## Citation
```

@misc{miranda2025cerradata4mmmultimodalbenchmarkdataset,
      title={CerraData-4MM: A multimodal benchmark dataset on Cerrado for land use and land cover classification},
      author={Mateus de Souza Miranda and Ronny Hänsch and Valdivino Alexandre de Santiago Júnior and Thales Sehn Körting and Erison Carlos dos Santos Monteiro},
      year={2025},
      eprint={2502.00083},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.00083},
}

```
---
## Contact
If you have any questions, please let us know at mateus.miranda@inpe.br



