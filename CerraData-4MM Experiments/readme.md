# How to access the trained models

The trained models are available at:

```
import kagglehub

# Download latest version
path = kagglehub.dataset_download("cerranet/trained-models-on-cerradata-4mm")

print("Path to dataset files:", path)
```

Both U-Net and TransNuSeg were trained in three primary scenarios: i) using both modalities, SAR and MSI; ii) using only MSI; and iii) using only SAR images. These scenarios were considered in two cases: a) data balancing with weights, and b) without weights.
As a result, you will find three directories (concat, msi, and sar) containing both models in each directory. 

---
