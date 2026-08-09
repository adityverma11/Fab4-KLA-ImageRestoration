import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. DATASET LOADER FOR .NPY FILES
# ==========================================
class SemiconductorDataset(Dataset):
    def __init__(self, noisy_dir, gt_dir=None, is_train=True):
        self.noisy_paths = sorted([os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir) if f.endswith('.npy')])
        
        if gt_dir:
            self.gt_paths = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith('.npy')])
        else:
            self.gt_paths = None
            
        self.is_train = is_train

    def __len__(self):
        return len(self.noisy_paths)

    def __getitem__(self, idx):
        noisy_img = np.load(self.noisy_paths[idx]).astype(np.float32)
        
        if noisy_img.ndim == 2:
            noisy_tensor = torch.from_numpy(noisy_img).unsqueeze(0) # (1, H, W)
        else:
            noisy_tensor = torch.from_numpy(noisy_img)

        if self.gt_paths:
            gt_img = np.load(self.gt_paths[idx]).astype(np.float32)
            if gt_img.ndim == 2:
                gt_tensor = torch.from_numpy(gt_img).unsqueeze(0) # (1, H, W)
            else:
                gt_tensor = torch.from_numpy(gt_img)
            return noisy_tensor, gt_tensor
        
        return noisy_tensor, os.path.basename(self.noisy_paths[idx])

# ==========================================
# 2. LIGHTWEIGHT NAFNet BLOCK FOR RESTORATION
# ==========================================
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c * 2, 3, padding=1)
        self.sg = SimpleGate()
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.norm1 = nn.GroupNorm(1, c)
        self.norm2 = nn.GroupNorm(1, c)
        self.mlp = nn.Sequential(
            nn.Conv2d(c, c * 2, 1),
            SimpleGate(),
            nn.Conv2d(c, c, 1)
        )

    def forward(self, x):
        res = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.sg(x)
        x = self.conv2(x)
        x = x + res

        res = x
        x = self.norm2(x)
        x = self.mlp(x)
        return x + res

class NAFNetSR(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, dim=32, num_blocks=4, scale=2):
        super().__init__()
        self.scale = scale
        self.intro = nn.Conv2d(in_channels, dim, 3, padding=1)
        self.blocks = nn.ModuleList([NAFBlock(dim) for _ in range(num_blocks)])
        self.upsample = nn.Sequential(
            nn.Conv2d(dim, dim * (scale ** 2), 3, padding=1),
            nn.PixelShuffle(scale)
        )
        self.outro = nn.Conv2d(dim, out_channels, 3, padding=1)

    def forward(self, x):
        res = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        
        x = self.intro(x)
        for block in self.blocks:
            x = block(x)
        x = self.upsample(x)
        x = self.outro(x)
        
        return x + res

# ==========================================
# 3. COMPOSITE LOSS
# ==========================================
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))

class CompositeLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.alpha = alpha

    def forward(self, pred, target):
        loss_pixel = self.charbonnier(pred, target)
        
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        loss_fft = F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))
        
        return loss_pixel + (self.alpha * loss_fft)

# ==========================================
# 4. FAST GPU TRAINING EXECUTION
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

    TRAIN_NOISY_DIR = r"D:\KLA_Hackathon\Data-public\train\train\NoisyLR"
    TRAIN_GT_DIR = r"D:\KLA_Hackathon\Data-public\train\train\GT"

    if os.path.exists(TRAIN_NOISY_DIR) and os.path.exists(TRAIN_GT_DIR):
        print("Paths verified successfully!")
        dataset = SemiconductorDataset(TRAIN_NOISY_DIR, TRAIN_GT_DIR)
        
        # Batch size set to 16 for RTX 3050 6GB VRAM
        loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)

        model = NAFNetSR(dim=32, num_blocks=4, scale=2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = CompositeLoss(alpha=0.1)
        scaler = torch.cuda.amp.GradScaler() # FP16 Mixed Precision Speedup

        print(f"Dataset Size: {len(dataset)} .npy image pairs found. Starting GPU training...\n")
        model.train()
        for epoch in range(1, 11): # 10 Epochs
            total_loss = 0.0
            for noisy, gt in loader:
                noisy, gt = noisy.to(device), gt.to(device)
                
                optimizer.zero_grad()
                
                # Autocast for Ampere GPU acceleration
                with torch.cuda.amp.autocast():
                    pred = model(noisy)
                    loss = criterion(pred, gt)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
            
            print(f"Epoch [{epoch}/10], Loss: {total_loss/len(loader):.6f}")

        torch.save(model.state_dict(), "nafnet_sr_best.pth")
        print("\nModel weights saved to 'nafnet_sr_best.pth' successfully!")
    else:
        print("[ERROR] Path mismatch! Double check folder locations.")