import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. MODEL DEFINITION
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
# 2. TEST DATASET LOADER
# ==========================================
class TestDataset(Dataset):
    def __init__(self, noisy_dir):
        self.noisy_paths = sorted([os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir) if f.endswith('.npy')])

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
# 3. FAST GPU INFERENCE EXECUTION
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Inference on: {device}")

    # Exact folder path detected from dir command
    TEST_NOISY_DIR = r"D:\KLA_Hackathon\Data-public\Test_NoisyLR\NoisyLR"
    OUTPUT_DIR = r"D:\KLA_Hackathon\Data-public\Restored_Test_Output"
    MODEL_WEIGHTS = "nafnet_sr_best.pth"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(TEST_NOISY_DIR) and os.path.exists(MODEL_WEIGHTS):
        print(f"Target test folder verified: {TEST_NOISY_DIR}")
        test_dataset = TestDataset(TEST_NOISY_DIR)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

        # Load weights safely
        model = NAFNetSR(dim=32, num_blocks=4, scale=2).to(device)
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True))
        model.eval()

        print(f"Found {len(test_dataset)} test files. Starting batch inference...")
        
        start_time = time.time()
        with torch.no_grad():
            for noisy_batch, filenames in test_loader:
                noisy_batch = noisy_batch.to(device)
                
                with torch.amp.autocast('cuda'):
                    restored_batch = model(noisy_batch)

                restored_np = restored_batch.cpu().numpy()
                for i in range(len(filenames)):
                    out_path = os.path.join(OUTPUT_DIR, filenames[i])
                    img_to_save = restored_np[i].squeeze()
                    np.save(out_path, img_to_save)

        total_time = time.time() - start_time
        print(f"\nInference completed in {total_time:.2f} seconds!")
        if len(test_dataset) > 0:
            print(f"Average speed: {total_time/len(test_dataset)*1000:.2f} ms/image")
        print(f"Restored files saved at: {OUTPUT_DIR}")
    else:
        print("[ERROR] Verification failed. Path or model weights missing.")