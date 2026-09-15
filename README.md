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
