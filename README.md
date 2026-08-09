# Fab4-KLA-ImageRestoration 🚀

**Team Name:** Fab4  
**Problem Statement:** SEM Image Denoising & 2x Super-Resolution (128x128 -> 256x256)  
**Event:** KLA Hackathon 2026  

---

## 📌 Project Overview
In semiconductor inspection, Scanning Electron Microscopy (SEM) images suffer from severe Poisson-Gaussian noise due to low electron-dose exposure. This repository contains **Team Fab4's** official submission: an end-to-end deep learning framework built on **NAFNet-SR** (*Nonlinear Activation-Free Network for Super-Resolution*).

Our model performs joint **Poisson-Gaussian noise reduction** and **2x spatial super-resolution** directly on raw binary floating-point `.npy` files with fast GPU inference throughput.

---

## 📦 Dataset Download
Due to file size limits, the raw dataset is hosted on Google Drive:
* 📁 **Dataset Link:** [Download SEM Dataset from Google Drive](https://drive.google.com/drive/folders/1VKiFW-kDk9-q5XRPu3nrl08OM94EwzV6?usp=sharing)

After downloading, extract the dataset folder inside the repository root as `Data-public/`:
```text
Fab4-KLA-ImageRestoration/
└── Data-public/
    └── train/
        ├── NoisyLR/
        └── CleanHR/