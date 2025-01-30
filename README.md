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

The dataset contains two hierarchical levels of classes: **Level 1 (L1)** and **Level 2 (L2)**. Below is the complete list of classes and their corresponding IDs:

| **L1**               | **ID** | **L2**                                | **ID** |
|-----------------------|--------|---------------------------------------|--------|
| Pasture              | 0      | Pasture                              | 0      |
| Arboreal             | 1      | Primary Natural Vegetation           | 1      |
| Agriculture          | 2      | Secondary Natural Vegetation         | 2      |
| Mining               | 3      | Water body                           | 3      |
| Building             | 4      | Mining                               | 4      |
| Water body           | 5      | Urban area                           | 5      |
| Other uses           | 6      | Other Built area                     | 6      |
|                      |        | Forestry                             | 7      |
|                      |        | Perennial Agriculture                | 8      |
|                      |        | Semi-perennial Agriculture           | 9      |
|                      |        | Temporary agriculture of 1 cycle     | 10     |
|                      |        | Temporary agriculture of 1 + cycle   | 11     |
|                      |        | Other Uses                           | 12     |
|                      |        | Deforestation 2022                   | 13     |


Text... 

![image](visual_setup/Head.png)

---

## How to Download

You can download the dataset using the `kagglehub` Python package. Below is an example of how to download the latest version of the dataset:

```python
import kagglehub

# Download the latest version
path = kagglehub.dataset_download("cerranet/cerradata-4mm")

print("Path to dataset files:", path)


---

## How to Download
