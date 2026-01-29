# 3DMoEST: 3D Mixture of Experts for Spatial Transcriptomics

A deep learning framework for predicting gene expression from 3D spatial transcriptomics data using morphology-guided mixture of experts.

## Overview

**3DMoEST** combines:
- **UNI**: A foundation model for histopathology image feature extraction (ViT-L/16 pretrained on 100M pathology images)
- **MoEST+**: A novel mixture-of-experts architecture that predicts gene expression using:
  - Visual features from histopathology images
  - 3D spatial coordinates with Fourier encoding
  - Morphological gradients (Sobel edge detection)
  - Sparse mixture-of-experts guided by tissue morphology

## Key Features

- **Morphology-Guided Expert Routing**: Automatically selects appropriate experts based on tissue morphology
- **3D Spatial Modeling**: Handles 3D spatial transcriptomics data with Fourier position encoding
- **Negative Binomial Loss**: Appropriate modeling of gene expression count data
- **Coupled Gradient Loss**: Enforces spatial smoothness while respecting tissue boundaries
- **Masked Spatial Masking (MSM)**: Improves robustness through visual feature masking

## Repository Structure


## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended: A100)
- 32GB+ RAM

### Dependencies

```bash
pip install torch torchvision
pip install timm h5py numpy scipy scikit-learn
pip install scanpy tqdm matplotlib seaborn
```

### UNI Model Setup

Download the UNI model weights:
```bash
# Follow instructions at: https://github.com/mahmoodlab/UNI
# Place model in: ./models/UNI/
```

## Quick Start

### 1. Data Preparation

Extract UNI features from histopathology images:

```bash
# Use Test:

# For HER2-ST dataset
python src/data_preparation/step2_extract_her2.py

# For MISAR dataset
python src/data_preparation/step2_extract_misar_fixed.py

# For openST dataset
python src/data_preparation/step2_extract_openst.py
```

**Input**: H5 files with `{patches, expression, coords_3d, gene_names}`
**Output**: H5 files with `{uni_features, sobel_gradients, expression, coords_3d, gene_names}`

### 2. Model Training

**K-fold cross-validation (recommended):**

```bash
# HER2-ST 5-fold CV
python src/training/train_her2_kfold.py --fold 0 --gpu 0
python src/training/train_her2_kfold.py --fold 1 --gpu 1
# ... repeat for folds 2-4

# MISAR 4D k-fold CV
python src/training/train_misar_4d_kfold.py --fold 0 --gpu 0
```

**Single 90/10 split:**

```bash
python src/training/train_moest_plus_final.py
```

### 3. Inference & Visualization

```bash
python src/inference/inference_her2_moest.py
```

Generates:
- Gene expression heatmaps (true vs predicted)
- Expert routing assignment maps
- Functional gradient visualizations


## Model Architecture


## Data Format

**H5 File Structure:**
```
dataset.h5
├── A1, B1, C1, ...         # Section names
│   ├── uni_features        # (N, 1024) - UNI embeddings
│   ├── coords_3d           # (N, 3) - 3D spatial coordinates
│   ├── sobel_gradients     # (N, H, W) - Morphological gradients
│   ├── expression          # (N, G) - Raw gene counts
│   └── patches             # (N, 224, 224, 3) - Image patches (optional)
└── gene_names              # (G,) - Gene name array
```

## Datasets

| Dataset | Type | Sections | Dimensions | Description |
|---------|------|----------|-----------|-------------|
| **HER2-ST** | 2D spatial | ~10 | 2D coords | Breast cancer HER2+ status |
| **MISAR** | 3D spatial | ~10 | 3D coords | Mouse embryo development |
| **openST** | 3D spatial | Multiple | 3D coords | General spatial transcriptomics |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{3dmoest2024,
  title={3DMoEST: Morphology-Guided Mixture of Experts for 3D Spatial Transcriptomics},
  author={Anonymous},
  journal={Under Review},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **UNI Model**: [Mahmood Lab](https://github.com/mahmoodlab/UNI)
- **Datasets**: HER2-ST, MISAR, openST consortiums
