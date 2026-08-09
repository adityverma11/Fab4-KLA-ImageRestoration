**Team:** Fab4  
**Event:** SEMI Hackathon 2026  
**Problem Statement:** AI-Based Restoration of Degraded Images for Semiconductor Inspection

## 🎯 Problem

Low-dose SEM imaging in semiconductor inspection introduces:
* **Poisson-Gaussian / speckle noise**
* **2× spatial downsampling**
* Loss of fine wafer-line and defect details

**Input:** Noisy LR `128×128` `.npy` image
**Target:** Clean HR `256×256` image

---
## 💡 Our Solution
We use a lightweight **NAFNet-SR** model for **joint denoising + 2× super-resolution**.
```text
Noisy LR (128×128)
        ↓
   Conv 3×3
        ↓
    NAFBlocks
(Simple Gate + SCA)
        ↓
 PixelShuffle ×2
        ↓
Restored HR (256×256)
```
The model is **activation-free**, lightweight, and optimized for fast GPU inference.

---

## 🧠 Loss Function
We combine spatial and frequency-domain reconstruction losses:
$$
L_{total} = L_{Charbonnier} + \lambda L_{FFT}
$$

* **Charbonnier Loss** → pixel-level restoration
* **FFT Loss** → preserves high-frequency SEM structures and fine details
---
## 📊 Verified Results

**Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU

| Metric              |                  Result |
| ------------------- | ----------------------: |
| Input → Output      |     `128×128 → 256×256` |
| Evaluation Speed    |      **20.63 ms/image** |
| 3,200 Images        |           **66.02 sec** |
| Batch Inference     |      **26.94 ms/image** |
| Model Size          |             **0.64 MB** |
| Tensor Keys         |                  **54** |
| Loss (Epoch 1 → 10) | **0.053853 → 0.036088** |
### Intensity Restoration

```text
Noisy Input : [0.00, 1.54]
Restored    : [0.01, 1.49]
Ground Truth: [0.00, 1.00]
```
---
## ▶️ Run the Project
### 1. Clone
```bash
git clone https://github.com/adityverma11/Fab4-KLA-ImageRestoration.git
cd Fab4-KLA-ImageRestoration
```
### 2. Install
```bash
pip install -r requirements.txt
```
### 3. Dataset
Download the SEM dataset:
**Google Drive:**
https://drive.google.com/drive/folders/1VKiFW-kDk9-q5XRPu3nrl08OM94EwzV6?usp=sharing
Place it as:
```text
Data-public/
└── train/
    ├── NoisyLR/
    └── CleanHR/
```
### 4. Run Evaluation
```bash
python evaluation.py \
  --input_dir Data-public/train/NoisyLR \
  --output_dir outputs
```
---
## 📁 Repository
```text
Fab4-KLA-ImageRestoration/
├── train.py              # Training
├── evaluation.py        # Evaluation & benchmarking
├── inference.py         # Batch inference
├── visualize.py         # Visualization & verification
├── nafnet_sr_best.pth   # Pre-trained weights (0.64 MB)
├── requirements.txt
└── outputs/
```
---
## 🔑 Key Highlights
**NAFNet-SR** • **Denoising + Super-Resolution** • **FFT Loss**
**0.64 MB Model** • **20.63 ms/image** • **Direct `.npy` Processing**

---

