import os
import random
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split, Subset



# CONFIG


SEED = 42

# YOUR EXACT DATASET PATHS
TRAIN_NOISY_DIR = r"D:\KLA_Hackathon\Data-public\train\train\NoisyLR"
TRAIN_GT_DIR = r"D:\KLA_Hackathon\Data-public\train\train\GT"

WEIGHTS_DIR = "weights"

BEST_MODEL_PATH = os.path.join(
    WEIGHTS_DIR,
    "best_model.pth"
)

LAST_MODEL_PATH = os.path.join(
    WEIGHTS_DIR,
    "last_model.pth"
)

# Training
EPOCHS = 60
BATCH_SIZE = 8

# Input = 128x128
# GT = 256x256
LR_PATCH_SIZE = 128
SCALE = 2

# Stable learning rate
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Model
DIM = 48
NUM_BLOCKS = 8

# Loss
FFT_WEIGHT = 0.05
SSIM_WEIGHT = 0.15

# Validation
VAL_RATIO = 0.20

NUM_WORKERS = 0


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DATASET
# ============================================================

class SemiconductorDataset(Dataset):

    def __init__(
        self,
        noisy_dir,
        gt_dir,
        patch_size=128,
        scale=2,
        training=True
    ):

        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir
        self.patch_size = patch_size
        self.scale = scale
        self.training = training

        noisy_files = {
            f for f in os.listdir(noisy_dir)
            if f.endswith(".npy")
        }

        gt_files = {
            f for f in os.listdir(gt_dir)
            if f.endswith(".npy")
        }

        common_files = sorted(
            noisy_files.intersection(gt_files)
        )

        if len(common_files) == 0:
            raise RuntimeError(
                "No matching .npy image pairs found."
            )

        self.filenames = common_files

        self.noisy_paths = [
            os.path.join(noisy_dir, f)
            for f in self.filenames
        ]

        self.gt_paths = [
            os.path.join(gt_dir, f)
            for f in self.filenames
        ]

        print(
            f"Matched image pairs: {len(self.filenames)}"
        )

    def __len__(self):
        return len(self.filenames)

    def _random_crop(self, noisy, gt):

        lr_h, lr_w = noisy.shape[-2:]
        gt_h, gt_w = gt.shape[-2:]

        expected_h = lr_h * self.scale
        expected_w = lr_w * self.scale

        if gt_h != expected_h or gt_w != expected_w:

            raise ValueError(
                f"Resolution mismatch:\n"
                f"NoisyLR: {noisy.shape}\n"
                f"GT: {gt.shape}\n"
                f"Expected GT: "
                f"{expected_h}x{expected_w}"
            )

        # If image itself is smaller than patch
        if (
            lr_h < self.patch_size
            or lr_w < self.patch_size
        ):
            return noisy, gt

        top = random.randint(
            0,
            lr_h - self.patch_size
        )

        left = random.randint(
            0,
            lr_w - self.patch_size
        )

        noisy = noisy[
            :,
            top:top + self.patch_size,
            left:left + self.patch_size
        ]

        gt_top = top * self.scale
        gt_left = left * self.scale

        gt_patch_size = (
            self.patch_size * self.scale
        )

        gt = gt[
            :,
            gt_top:gt_top + gt_patch_size,
            gt_left:gt_left + gt_patch_size
        ]

        return noisy, gt

    def _augment(self, noisy, gt):

        # Horizontal flip
        if random.random() < 0.5:

            noisy = torch.flip(
                noisy,
                dims=[2]
            )

            gt = torch.flip(
                gt,
                dims=[2]
            )

        # Vertical flip
        if random.random() < 0.5:

            noisy = torch.flip(
                noisy,
                dims=[1]
            )

            gt = torch.flip(
                gt,
                dims=[1]
            )

        # 0 / 90 / 180 / 270 degree rotation
        k = random.randint(0, 3)

        if k > 0:

            noisy = torch.rot90(
                noisy,
                k,
                dims=[1, 2]
            )

            gt = torch.rot90(
                gt,
                k,
                dims=[1, 2]
            )

        return noisy, gt

    def __getitem__(self, idx):

        noisy = np.load(
            self.noisy_paths[idx]
        ).astype(np.float32)

        gt = np.load(
            self.gt_paths[idx]
        ).astype(np.float32)

        if noisy.ndim == 2:
            noisy = noisy[None, :, :]

        if gt.ndim == 2:
            gt = gt[None, :, :]

        noisy = torch.from_numpy(noisy)
        gt = torch.from_numpy(gt)

        if noisy.shape[0] != 1:
            raise ValueError(
                f"Expected 1-channel noisy image, "
                f"got {noisy.shape}"
            )

        if gt.shape[0] != 1:
            raise ValueError(
                f"Expected 1-channel GT image, "
                f"got {gt.shape}"
            )

        # Safety check
        if not torch.isfinite(noisy).all():
            raise ValueError(
                f"NaN/Inf found in noisy image: "
                f"{self.filenames[idx]}"
            )

        if not torch.isfinite(gt).all():
            raise ValueError(
                f"NaN/Inf found in GT image: "
                f"{self.filenames[idx]}"
            )

        if self.training:

            noisy, gt = self._random_crop(
                noisy,
                gt
            )

            noisy, gt = self._augment(
                noisy,
                gt
            )

        return noisy, gt


# ============================================================
# SIMPLE GATE
# ============================================================

class SimpleGate(nn.Module):

    def forward(self, x):

        x1, x2 = x.chunk(2, dim=1)

        return x1 * x2


# ============================================================
# STABLE NAF BLOCK
# ============================================================

class NAFBlock(nn.Module):

    def __init__(self, c):

        super().__init__()

        self.norm1 = nn.GroupNorm(
            1,
            c
        )

        self.conv1 = nn.Conv2d(
            c,
            c * 2,
            kernel_size=3,
            padding=1
        )

        self.sg = SimpleGate()

        self.conv2 = nn.Conv2d(
            c,
            c,
            kernel_size=3,
            padding=1
        )

        self.norm2 = nn.GroupNorm(
            1,
            c
        )

        self.mlp = nn.Sequential(

            nn.Conv2d(
                c,
                c * 2,
                kernel_size=1
            ),

            SimpleGate(),

            nn.Conv2d(
                c,
                c,
                kernel_size=1
            )
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Learnable residual scaling.
        # Starts at zero -> stable training.
        # ----------------------------------------------------

        self.beta = nn.Parameter(
            torch.zeros(
                1,
                c,
                1,
                1
            )
        )

        self.gamma = nn.Parameter(
            torch.zeros(
                1,
                c,
                1,
                1
            )
        )

    def forward(self, x):

        # First residual branch
        residual = x

        y = self.norm1(x)
        y = self.conv1(y)
        y = self.sg(y)
        y = self.conv2(y)

        x = residual + (
            y * self.beta
        )

        # Second residual branch
        residual = x

        y = self.norm2(x)
        y = self.mlp(y)

        x = residual + (
            y * self.gamma
        )

        return x


# ============================================================
# NAF-STYLE SUPER RESOLUTION MODEL
# ============================================================

class NAFNetSR(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        dim=48,
        num_blocks=8,
        scale=2
    ):

        super().__init__()

        self.scale = scale

        self.intro = nn.Conv2d(
            in_channels,
            dim,
            kernel_size=3,
            padding=1
        )

        self.blocks = nn.Sequential(
            *[
                NAFBlock(dim)
                for _ in range(num_blocks)
            ]
        )

        self.middle = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            padding=1
        )

        self.upsample = nn.Sequential(

            nn.Conv2d(
                dim,
                dim * (scale ** 2),
                kernel_size=3,
                padding=1
            ),

            nn.PixelShuffle(scale)
        )

        self.outro = nn.Conv2d(
            dim,
            out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        # Bilinear base image
        base = F.interpolate(
            x,
            scale_factor=self.scale,
            mode="bilinear",
            align_corners=False
        )

        features = self.intro(x)

        features = self.blocks(
            features
        )

        features = self.middle(
            features
        )

        features = self.upsample(
            features
        )

        residual = self.outro(
            features
        )

        output = base + residual

        return output


# ============================================================
# CHARBONNIER LOSS
# ============================================================

class CharbonnierLoss(nn.Module):

    def __init__(self, eps=1e-3):

        super().__init__()

        self.eps = eps

    def forward(
        self,
        prediction,
        target
    ):

        diff = prediction - target

        loss = torch.sqrt(
            diff * diff +
            self.eps * self.eps
        )

        return loss.mean()


# ============================================================
# SSIM
# ============================================================

def create_gaussian_window(
    window_size,
    sigma,
    device
):

    coords = torch.arange(
        window_size,
        dtype=torch.float32,
        device=device
    )

    coords -= window_size // 2

    gaussian = torch.exp(
        -(coords ** 2) /
        (2 * sigma ** 2)
    )

    gaussian = (
        gaussian /
        gaussian.sum()
    )

    window = (
        gaussian[:, None] *
        gaussian[None, :]
    )

    return window


def calculate_ssim(
    img1,
    img2,
    window_size=11,
    sigma=1.5
):

    # IMPORTANT:
    # Always calculate SSIM in FP32.
    img1 = img1.float()
    img2 = img2.float()

    channel = img1.size(1)

    window = create_gaussian_window(
        window_size,
        sigma,
        img1.device
    )

    window = window.expand(
        channel,
        1,
        window_size,
        window_size
    )

    padding = window_size // 2

    mu1 = F.conv2d(
        img1,
        window,
        padding=padding,
        groups=channel
    )

    mu2 = F.conv2d(
        img2,
        window,
        padding=padding,
        groups=channel
    )

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2

    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(
            img1 * img1,
            window,
            padding=padding,
            groups=channel
        )
        - mu1_sq
    )

    sigma2_sq = (
        F.conv2d(
            img2 * img2,
            window,
            padding=padding,
            groups=channel
        )
        - mu2_sq
    )

    sigma12 = (
        F.conv2d(
            img1 * img2,
            window,
            padding=padding,
            groups=channel
        )
        - mu1_mu2
    )

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    numerator = (
        (2 * mu1_mu2 + C1)
        *
        (2 * sigma12 + C2)
    )

    denominator = (
        (mu1_sq + mu2_sq + C1)
        *
        (sigma1_sq + sigma2_sq + C2)
    )

    score = numerator / (
        denominator + 1e-8
    )

    return score.mean()


class SSIMLoss(nn.Module):

    def forward(
        self,
        prediction,
        target
    ):

        return 1.0 - calculate_ssim(
            prediction,
            target
        )


# ============================================================
# FREQUENCY LOSS
# ============================================================

class FrequencyLoss(nn.Module):

    def forward(
        self,
        prediction,
        target
    ):

        # IMPORTANT:
        # FFT MUST be FP32.
        prediction = prediction.float()
        target = target.float()

        pred_fft = torch.fft.rfft2(
            prediction,
            norm="ortho"
        )

        target_fft = torch.fft.rfft2(
            target,
            norm="ortho"
        )

        pred_mag = torch.abs(
            pred_fft
        )

        target_mag = torch.abs(
            target_fft
        )

        return F.l1_loss(
            pred_mag,
            target_mag
        )


# ============================================================
# COMPOSITE LOSS
# ============================================================

class CompositeLoss(nn.Module):

    def __init__(
        self,
        fft_weight=0.05,
        ssim_weight=0.15
    ):

        super().__init__()

        self.charbonnier = (
            CharbonnierLoss()
        )

        self.frequency = (
            FrequencyLoss()
        )

        self.ssim = (
            SSIMLoss()
        )

        self.fft_weight = fft_weight
        self.ssim_weight = ssim_weight

    def forward(
        self,
        prediction,
        target
    ):

        # ALL LOSS COMPUTATION IN FP32
        prediction = prediction.float()
        target = target.float()

        # For loss only.
        # We do NOT modify noisy input.
        prediction_for_loss = torch.clamp(
            prediction,
            0.0,
            1.0
        )

        loss_pixel = (
            self.charbonnier(
                prediction_for_loss,
                target
            )
        )

        loss_fft = (
            self.frequency(
                prediction_for_loss,
                target
            )
        )

        loss_ssim = (
            self.ssim(
                prediction_for_loss,
                target
            )
        )

        total_loss = (
            loss_pixel
            +
            self.fft_weight * loss_fft
            +
            self.ssim_weight * loss_ssim
        )

        return total_loss


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(
    prediction,
    target
):

    prediction = prediction.float()
    target = target.float()

    prediction = torch.clamp(
        prediction,
        0.0,
        1.0
    )

    target = torch.clamp(
        target,
        0.0,
        1.0
    )

    mse = F.mse_loss(
        prediction,
        target
    )

    mse_value = mse.item()

    if (
        not np.isfinite(mse_value)
        or mse_value <= 1e-12
    ):
        return 0.0

    psnr = (
        10.0 *
        torch.log10(
            1.0 / mse
        )
    )

    return psnr.item()


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    count = 0

    for noisy, gt in loader:

        noisy = noisy.to(
            device,
            non_blocking=True
        )

        gt = gt.to(
            device,
            non_blocking=True
        )

        # Validation forward can use AMP
        # but loss is explicitly FP32.
        if device.type == "cuda":

            with torch.amp.autocast(
                "cuda"
            ):

                prediction = model(
                    noisy
                )

        else:

            prediction = model(
                noisy
            )

        prediction = prediction.float()

        loss = criterion(
            prediction,
            gt.float()
        )

        if not torch.isfinite(loss):

            return (
                float("inf"),
                0.0,
                0.0
            )

        prediction = torch.clamp(
            prediction,
            0.0,
            1.0
        )

        total_loss += loss.item()

        total_psnr += calculate_psnr(
            prediction,
            gt
        )

        total_ssim += calculate_ssim(
            prediction,
            gt
        ).item()

        count += 1

    return (
        total_loss / max(count, 1),
        total_psnr / max(count, 1),
        total_ssim / max(count, 1)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "KLA HACKATHON - STABLE IMAGE RESTORATION TRAINING"
    )
    print("=" * 70)

    set_seed(SEED)

    os.makedirs(
        WEIGHTS_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    else:

        device = torch.device(
            "cpu"
        )

        print(
            "WARNING: CUDA not available."
        )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # CHECK DATASET PATH
    # --------------------------------------------------------

    if not os.path.isdir(
        TRAIN_NOISY_DIR
    ):

        raise FileNotFoundError(
            f"NoisyLR folder not found:\n"
            f"{TRAIN_NOISY_DIR}"
        )

    if not os.path.isdir(
        TRAIN_GT_DIR
    ):

        raise FileNotFoundError(
            f"GT folder not found:\n"
            f"{TRAIN_GT_DIR}"
        )

    print(
        "\nDataset paths verified."
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    full_dataset = SemiconductorDataset(
        TRAIN_NOISY_DIR,
        TRAIN_GT_DIR,
        patch_size=LR_PATCH_SIZE,
        scale=SCALE,
        training=True
    )

    total_size = len(
        full_dataset
    )

    val_size = max(
        1,
        int(total_size * VAL_RATIO)
    )

    train_size = (
        total_size -
        val_size
    )

    generator = torch.Generator().manual_seed(
        SEED
    )

    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator
    )

    # --------------------------------------------------------
    # VALIDATION DATASET
    # NO RANDOM CROP / AUGMENTATION
    # --------------------------------------------------------

    validation_dataset = SemiconductorDataset(
        TRAIN_NOISY_DIR,
        TRAIN_GT_DIR,
        patch_size=LR_PATCH_SIZE,
        scale=SCALE,
        training=False
    )

    val_dataset = Subset(
        validation_dataset,
        val_subset.indices
    )

    print("\nDataset split:")
    print(
        f"Total      : {total_size}"
    )
    print(
        f"Training   : {train_size}"
    )
    print(
        f"Validation : {val_size}"
    )

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        )
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = NAFNetSR(
        in_channels=1,
        out_channels=1,
        dim=DIM,
        num_blocks=NUM_BLOCKS,
        scale=SCALE
    ).to(device)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print("\nModel:")
    print(
        f"Dimension      : {DIM}"
    )
    print(
        f"NAF blocks     : {NUM_BLOCKS}"
    )
    print(
        f"Scale          : x{SCALE}"
    )
    print(
        f"Parameters     : {total_parameters:,}"
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = CompositeLoss(
        fft_weight=FFT_WEIGHT,
        ssim_weight=SSIM_WEIGHT
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    if device.type == "cuda":

        scaler = torch.amp.GradScaler(
            "cuda"
        )

    else:

        scaler = None

    # --------------------------------------------------------
    # BEST
    # --------------------------------------------------------

    best_psnr = -float("inf")
    best_ssim = -float("inf")

    print("\nStarting training...")
    print("=" * 70)

    training_start = time.time()

    # ========================================================
    # EPOCHS
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        epoch_loss = 0.0

        epoch_start = time.time()

        valid_batches = 0

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        for noisy, gt in train_loader:

            noisy = noisy.to(
                device,
                non_blocking=True
            )

            gt = gt.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------------
            # FORWARD
            # ------------------------------------------------

            if device.type == "cuda":

                with torch.amp.autocast(
                    "cuda"
                ):

                    prediction = model(
                        noisy
                    )

                # IMPORTANT:
                # Convert prediction to FP32
                # BEFORE FFT / SSIM / loss.
                prediction = prediction.float()

                loss = criterion(
                    prediction,
                    gt.float()
                )

            else:

                prediction = model(
                    noisy
                )

                prediction = prediction.float()

                loss = criterion(
                    prediction,
                    gt.float()
                )

            # ------------------------------------------------
            # NaN CHECK
            # ------------------------------------------------

            if not torch.isfinite(loss):

                print(
                    "\n[WARNING] Non-finite loss detected!"
                )

                print(
                    f"Epoch: {epoch}"
                )

                print(
                    "Skipping this batch."
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                continue

            # ------------------------------------------------
            # BACKPROP
            # ------------------------------------------------

            if device.type == "cuda":

                scaler.scale(
                    loss
                ).backward()

                # Unscale before clipping
                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=0.5
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=0.5
                )

                optimizer.step()

            epoch_loss += loss.item()

            valid_batches += 1

        # ----------------------------------------------------
        # TRAIN LOSS
        # ----------------------------------------------------

        if valid_batches == 0:

            print(
                "ERROR: No valid batches in epoch."
            )

            break

        train_loss = (
            epoch_loss /
            valid_batches
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_loss, val_psnr, val_ssim = validate(
            model,
            val_loader,
            criterion,
            device
        )

        # Scheduler
        if np.isfinite(val_psnr):

            scheduler.step(
                val_psnr
            )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        epoch_time = (
            time.time() -
            epoch_start
        )

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"PSNR: {val_psnr:.4f} dB | "
            f"SSIM: {val_ssim:.6f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        # ----------------------------------------------------
        # SAVE BEST
        # ----------------------------------------------------

        if (
            np.isfinite(val_psnr)
            and val_psnr > best_psnr
        ):

            best_psnr = val_psnr
            best_ssim = val_ssim

            checkpoint = {

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "epoch":
                    epoch,

                "best_psnr":
                    best_psnr,

                "best_ssim":
                    best_ssim,

                "config": {

                    "in_channels": 1,

                    "out_channels": 1,

                    "dim": DIM,

                    "num_blocks":
                        NUM_BLOCKS,

                    "scale": SCALE,

                    "patch_size":
                        LR_PATCH_SIZE,

                    "batch_size":
                        BATCH_SIZE,

                    "learning_rate":
                        LEARNING_RATE,

                    "weight_decay":
                        WEIGHT_DECAY,

                    "fft_weight":
                        FFT_WEIGHT,

                    "ssim_weight":
                        SSIM_WEIGHT,

                    "seed": SEED
                }
            }

            torch.save(
                checkpoint,
                BEST_MODEL_PATH
            )

            print(
                f">>> BEST MODEL SAVED "
                f"(PSNR={best_psnr:.4f} dB, "
                f"SSIM={best_ssim:.6f})"
            )

    # --------------------------------------------------------
    # LAST MODEL
    # --------------------------------------------------------

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "config": {

                "in_channels": 1,
                "out_channels": 1,
                "dim": DIM,
                "num_blocks": NUM_BLOCKS,
                "scale": SCALE,
                "patch_size":
                    LR_PATCH_SIZE
            }
        },
        LAST_MODEL_PATH
    )

    total_time = (
        time.time() -
        training_start
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best PSNR : {best_psnr:.4f} dB"
    )

    print(
        f"Best SSIM : {best_ssim:.6f}"
    )

    print(
        f"Training Time : "
        f"{total_time / 60:.2f} minutes"
    )

    print(
        f"\nBEST MODEL:"
    )

    print(
        BEST_MODEL_PATH
    )

    print(
        "\nDo NOT train on the hidden test dataset."
    )

    print(
        "Use the trained checkpoint only for inference."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()