# Fab4-KLA-ImageRestoration 🚀

**Team Name:** Fab4  
**Problem Statement:** AI-Based Restoration of Degraded Images for Semiconductor Inspection  
**Event:** KLA Hackathon 2026  

---

## ⚡ Quick Start (Run Inference in 3 Steps)

Reviewers can clone this repository and execute evaluation/inference out-of-the-box using pre-trained weights (`nafnet_sr_best.pth`).

### Step 1: Clone Repository
```bash
git clone [https://github.com/adityverma11/Fab4-KLA-ImageRestoration.git](https://github.com/adityverma11/Fab4-KLA-ImageRestoration.git)
cd Fab4-KLA-ImageRestoration
Step 2: Install Dependencies
pip install -r requirements.txt

Step 3: Run Evaluation Script
Run automated benchmarking on any input directory containing test .npy images:
python evaluation.py --input_dir Data-public/train/train/NoisyLR --output_dir outputs

📌 Project OverviewIn semiconductor manufacturing, Scanning Electron Microscopy (SEM) inspection images are vital for sub-micron defect detection. However, low-dose electron beam imaging introduces severe physical degradations:Speckle & Poisson-Gaussian Noise: Grainy pixel distortion pushing pixel values beyond the standard $[0, 1]$ normalized range (observed up to $1.54$).Spatial Down-sampling ($2\times$ Detail Loss): Resolution reduction from $128 \times 128 \to 256 \times 256$, obscuring fine wafer line structures.Team Fab4's Solution utilizes NAFNet-SR (Nonlinear Activation-Free Network for Super-Resolution), an ultra-lightweight ($0.64\text{ MB}$) architecture. It performs joint Poisson-Gaussian denoising and $2\times$ spatial super-resolution directly on raw binary floating-point .npy files with real-time GPU inference throughput.

🔬 Model Architecture & Technical Pipeline1. Mathematical FormulationThe degradation pipeline is modeled as:$$y = f(x) = \mathcal{D}_{2\times}(\mathcal{N}_{\text{Poisson-Gaussian}}(x))$$Where:$x \in [0, 1]$ represents the Clean High-Resolution Ground Truth image ($256 \times 256$).$y \in [0, 1.54]$ represents the Noisy Low-Resolution input image ($128 \times 128$).Reconstructed output $\hat{x} = \mathcal{G}_{\theta}(y)$ recovers spatial details back to $256 \times 256$.

2. NAFNet-SR Architecture Details
Eliminates traditional non-linear activations (GELU/ReLU) in favor of a Simple Gate (element-wise multiplication) and Simplified Channel Attention (SCA) across 54 tensor parameter blocks to optimize GPU memory and latency:

Input Noisy LR (128x128)
       │
       ▼
┌──────────────┐
│ Conv2d 3x3   │ ──► Initial Feature Extraction (C Channels)
└──────────────┘
       │
       ▼
┌──────────────┐
│ NAFBlocks    │ ──► LayerNorm ➔ Conv ➔ Simple Gate ➔ SCA ➔ Dropout
└──────────────┘
       │
       ▼
┌──────────────┐
│ PixelShuffle │ ──► 2x Sub-Pixel Upsampling (256x256)
└──────────────┘
       │
       ▼
Restored Output (256x256)

3. Loss Function Strategy$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{Charbonnier}}(\hat{x}, x) + \lambda \cdot \mathcal{L}_{\text{FFT}}(\hat{x}, x)$$

📊 Live System Benchmarks & Verified Test Metrics
All metrics recorded directly from GPU terminal execution (NVIDIA GeForce RTX 3050 6GB Laptop GPU / CUDA):

Metric / Parameter,Degraded Input (NoisyLR),Restored Output (Fab4 NAFNet-SR),Target Ground Truth
Spatial Dimensions,128×128,256×256,256×256
Intensity Range,"[0.00,1.54] (Corrupted)","[0.01,1.49] (Restored)","[0.00,1.00]"
Evaluation Speed (evaluation.py),—,"20.63 ms / image (3,200 files in 66.02 s)",Real-time GPU benchmark
Batch Inference Speed (inference.py),—,26.94 ms / image (400 files in 10.77 s),High efficiency
Model Weight Checkpoint Size,—,0.64 MB (54 Tensor Keys),Ultra-compact
Training Loss Convergence,0.053853 (Epoch 1),0.036088 (Epoch 10),Stable convergence

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
🏗️ Repository Structure
Fab4-KLA-ImageRestoration/
├── evaluation.py            # Standalone evaluation script for automated benchmarking
├── train.py                 # Reproducible training script with FP16 support
├── inference.py             # Inference script for batch processing
├── visualize.py             # Script to verify shape & range metrics
├── requirements.txt         # Environment dependencies
├── README.md                # Official project documentation
├── nafnet_sr_best.pth       # Trained model weights checkpoint (0.64 MB)
└── outputs/                 # Restored output .npy files directory
