import os
import argparse
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. MODEL DEFINITION (EXACT MATCH)
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
# 2. DATASET LOADER FOR EVALUATION
# ==========================================
class TestDataset(Dataset):
    def __init__(self, noisy_paths):
        self.noisy_paths = noisy_paths

    def __len__(self):
        return len(self.noisy_paths)

    def __getitem__(self, idx):
        file_path = self.noisy_paths[idx]
        noisy_img = np.load(file_path).astype(np.float32)
        if noisy_img.ndim == 2:
            noisy_tensor = torch.from_numpy(noisy_img).unsqueeze(0)
        else:
            noisy_tensor = torch.from_numpy(noisy_img)
            
        return noisy_tensor, os.path.basename(file_path)

# ==========================================
# 3. STANDALONE EVALUATION SCRIPT
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="KLA Benchmarking Evaluation Script - Team Fab4")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input test directory containing .npy files")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save restored .npy files")
    parser.add_argument("--weights", type=str, default="nafnet_sr_best.pth", help="Path to model weights file")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Team Fab4] Running evaluation on device: {device}")

    # Recursive check for .npy files inside input_dir
    noisy_paths = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.npy"), recursive=True))
    if not noisy_paths:
        noisy_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.npy")))

    if not noisy_paths:
        print(f"[Error] No .npy files found in path: {args.input_dir}")
        return

    print(f"[Team Fab4] Found {len(noisy_paths)} test files in '{args.input_dir}'.")

    # Load Model Weights
    model = NAFNetSR(dim=32, num_blocks=4, scale=2).to(device)
    if os.path.exists(args.weights):
        try:
            model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
        except TypeError:
            model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"[Team Fab4] Weights successfully loaded from '{args.weights}'")
    else:
        print(f"[Error] Weights file '{args.weights}' missing!")
        return

    model.eval()

    test_dataset = TestDataset(noisy_paths)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    print("Starting batch evaluation...")
    start_time = time.time()

    with torch.no_grad():
        for noisy_batch, filenames in test_loader:
            noisy_batch = noisy_batch.to(device)
            
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    restored_batch = model(noisy_batch)
            else:
                restored_batch = model(noisy_batch)

            restored_np = restored_batch.cpu().numpy()
            for i in range(len(filenames)):
                out_path = os.path.join(args.output_dir, filenames[i])
                img_to_save = restored_np[i].squeeze()
                np.save(out_path, img_to_save)

    total_time = time.time() - start_time
    avg_speed = (total_time / len(test_dataset)) * 1000

    print(f"\n[Benchmarking Complete]")
    print(f"- Total Time: {total_time:.2f} seconds")
    print(f"- Average Speed: {avg_speed:.2f} ms/image")
    print(f"- Restored files saved at: {args.output_dir}")

if __name__ == "__main__":
    main()