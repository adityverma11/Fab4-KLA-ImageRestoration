# Fab4 – AI Image Restoration for Semiconductor Inspection

**Team:** Fab4
**Event:** SEMI Hackathon 2026
**Problem Statement:** AI-Based Restoration of Degraded Images for Semiconductor Inspection

---

## 🎯 Problem

Restore noisy, low-resolution SEM images while preserving important semiconductor structures and defect details.

**Input:** Noisy LR `128×128` single-channel `.npy` image
**Target:** Restored HR `256×256` single-channel `.npy` image

---

## 💡 Our Solution

We use a custom **NAFNet-style Super-Resolution (NAFNet-SR)** model for **joint denoising + 2× super-resolution**.

```
Noisy LR (128×128)
        ↓
   Input Conv
        ↓
   NAF Blocks ×8
        ↓
   Middle Conv
        ↓
 PixelShuffle ×2
        ↓
   Output Conv
        ↓
 Global Residual
        ↓
Restored HR (256×256)
```

The model is lightweight and designed for fast GPU inference.

---

## Model Configuration

| Parameter         |   Value |
| ----------------- | ------: |
| Feature Dimension |      48 |
| NAF Blocks        |       8 |
| Upscaling Factor  |      ×2 |
| Model Parameters  | 662,401 |
| Input             | 128×128 |
| Output            | 256×256 |

---

## 🧠 Loss Function

We use a combined reconstruction loss:

```
L = Charbonnier + 0.05 × FFT + 0.15 × SSIM
```

- **Charbonnier Loss** → pixel-level restoration
- **FFT Loss** → frequency-domain reconstruction
- **SSIM Loss** → structural similarity

---

## Training Configuration

| Parameter     | Value |
| ------------- | ----: |
| Optimizer     | AdamW |
| Learning Rate |  1e-4 |
| Batch Size    |     8 |
| Epochs        |    60 |

Training includes random cropping, horizontal/vertical flips, and 90° rotations.

---

## 📊 Verified Results

**Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU

| Metric            |                Result |
| ----------------- | --------------------: |
| Test Images       |               **400** |
| Outputs Generated |               **400** |
| Input → Output    | **128×128 → 256×256** |
| Model Parameters  |           **662,401** |
| Inference Time    |        **10.278 sec** |
| Average           |   **25.695 ms/image** |
| Throughput        |  **38.92 images/sec** |

400/400 outputs were successfully generated.

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

For the tested CUDA 12.1 setup:

```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Dataset

Download the SEM dataset:

**Google Drive:** [https://drive.google.com/drive/folders/1VKiFW-kDk9-q5XRPu3nrl08OM94EwzV6?usp=sharing](https://drive.google.com/drive/folders/1VKiFW-kDk9-q5XRPu3nrl08OM94EwzV6?usp=sharing)

Place it as:

```
Data-public/
├── train/
│   ├── NoisyLR/
│   └── CleanHR/
│
└── Test_NoisyLR/
    └── NoisyLR/
```

### 4. Run Evaluation

```bash
python evaluation.py ^
  --input_dir "Data-public\Test_NoisyLR\NoisyLR" ^
  --output_dir "Restored_Test_Output" ^
  --weights "weights\best_model.pth"
```

---

## 📁 Repository

```
Fab4-KLA-ImageRestoration/
├── train.py
├── evaluation.py
├── inference.py
├── visualize_results.py
├── requirements.txt
├── weights/
│   └── best_model.pth
├── Data-public/
│   ├── train/
│   │   ├── NoisyLR/
│   │   └── CleanHR/
│   └── Test_NoisyLR/
│       └── NoisyLR/
├── Restored_Test_Output/
└── results/
```

---

## 🔑 Key Highlights

**NAFNet-SR** • Denoising + 2× Super-Resolution • Charbonnier + FFT + SSIM

**662K Parameters** • **400/400 Successful Outputs** • **38.92 Images/sec** • **25.695 ms/image** • Direct `.npy` Processing
