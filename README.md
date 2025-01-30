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

| **L1 Class**      | **L1 ID** | **L2 Class**                                | **L2 ID** | **RGB Mask**   |
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

## How to use

### Data downloading
You can download the dataset using the `kagglehub` Python package. Below is an example of how to download the latest version of the dataset:

```python
import kagglehub

# Download the latest version
path = kagglehub.dataset_download("cerranet/cerradata-4mm")

print("Path to dataset files:", path)

```


### Data loading 

```python
import ...

```


### Training using U-Net 
```python
import ...

```

---
## Default scores

---
## Reference it


