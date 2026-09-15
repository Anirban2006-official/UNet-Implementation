# Biomedical Image Segmentation using U-Net in PyTorch

This repository contains a PyTorch implementation of the **U-Net** architecture for semantic segmentation on biomedical datasets (e.g., ISBI challenge dataset). The pipeline includes Google Drive integration, GPU acceleration, custom dataset handling, and model training.

---

## 📌 Project Overview

Medical image segmentation requires precise pixel-level localization along with high-level context. The **U-Net** architecture achieves this through an encoder-decoder framework connected via skip connections:
* **Contracting Path (Encoder):** Captures multi-scale contextual features using convolutional layers and max pooling.
* **Expanding Path (Decoder):** Restores spatial resolution via transposed convolutions, concatenating high-resolution feature maps from the encoder path.

---

## 🛠️ Requirements & Setup

### Dependencies
Ensure you have the following installed in your environment (or run inside Google Colab):

```bash
pip install torch torchvision numpy matplotlib pillow scipy
```

### Hardware Acceleration
This project automatically detects and utilizes CUDA-enabled GPUs:
```bash
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```
### 📁 Dataset Preparation
Upload your compressed dataset (archive.zip / ISBI.zip) to Google Drive.
Mount Drive and extract the files directly to your session storage:

```bash
import os
import zipfile
from google.colab import drive

drive.mount('/content/drive')

ZIP_FILE_PATH = '/content/drive/MyDrive/ISBI/archive.zip'
EXTRACT_DIR = '/content/dataset/'

os.makedirs(EXTRACT_DIR, exist_ok=True)
with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

print("Dataset extracted successfully.")
```

### Architecture Design

```
Input Image (1xHxW) ──> [Encoder Conv Blocks] ──Downsample──> Bottleneck
                             │                                    │
                             └─── Skip Connections ─── Upsample ──┘
                                                          │
                                                [Decoder Conv Blocks] ──> Output Mask (1xHxW)
```

