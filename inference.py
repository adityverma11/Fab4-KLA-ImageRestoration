import os
import glob
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# KLA HACKATHON 2026
# FINAL INFERENCE PIPELINE
#
# SAME ARCHITECTURE AS TRAINING
#
# Input  : 1 x 128 x 128
# Output : 1 x 256 x 256
#
# Feature dimension : 48
# NAF blocks        : 8
# Upscaling         : x2
#
# IMPORTANT:
# This script ONLY performs inference.
# It does NOT train or modify the model.
# ============================================================


# ============================================================
# 1. SIMPLE GATE
# ============================================================

class SimpleGate(nn.Module):

    def forward(self, x):

        x1, x2 = x.chunk(2, dim=1)

        return x1 * x2


# ============================================================
# 2. STABLE NAF BLOCK
# EXACT SAME AS TRAINING
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

        # IMPORTANT:
        # These two parameters MUST exist because
        # they were learned during training.

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

        # ----------------------------------------------------
        # First residual branch
        # ----------------------------------------------------

        residual = x

        y = self.norm1(x)

        y = self.conv1(y)

        y = self.sg(y)

        y = self.conv2(y)

        x = residual + (
            y * self.beta
        )

        # ----------------------------------------------------
        # Second residual branch
        # ----------------------------------------------------

        residual = x

        y = self.norm2(x)

        y = self.mlp(y)

        x = residual + (
            y * self.gamma
        )

        return x


# ============================================================
# 3. NAFNet SUPER-RESOLUTION MODEL
# EXACT SAME AS TRAINING
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

        # ----------------------------------------------------
        # Input projection
        # ----------------------------------------------------

        self.intro = nn.Conv2d(
            in_channels,
            dim,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # NAF blocks
        # ----------------------------------------------------

        self.blocks = nn.Sequential(
            *[
                NAFBlock(dim)
                for _ in range(num_blocks)
            ]
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # This layer exists in the trained model.
        # ----------------------------------------------------

        self.middle = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            padding=1
        )

        # ----------------------------------------------------
        # x2 Super Resolution
        # ----------------------------------------------------

        self.upsample = nn.Sequential(

            nn.Conv2d(
                dim,
                dim * (scale ** 2),
                kernel_size=3,
                padding=1
            ),

            nn.PixelShuffle(scale)
        )

        # ----------------------------------------------------
        # Output projection
        # ----------------------------------------------------

        self.outro = nn.Conv2d(
            dim,
            out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Bilinear base image
        # ----------------------------------------------------

        base = F.interpolate(
            x,
            scale_factor=self.scale,
            mode="bilinear",
            align_corners=False
        )

        # ----------------------------------------------------
        # Main network
        # ----------------------------------------------------

        features = self.intro(x)

        features = self.blocks(
            features
        )

        # IMPORTANT:
        # Same middle layer used during training

        features = self.middle(
            features
        )

        # ----------------------------------------------------
        # Super Resolution
        # ----------------------------------------------------

        features = self.upsample(
            features
        )

        residual = self.outro(
            features
        )

        # ----------------------------------------------------
        # Global residual learning
        # ----------------------------------------------------

        output = base + residual

        return output


# ============================================================
# 4. TEST DATASET
# ============================================================

class TestDataset(Dataset):

    def __init__(self, file_paths):

        self.file_paths = file_paths

    def __len__(self):

        return len(self.file_paths)

    def __getitem__(self, idx):

        file_path = self.file_paths[idx]

        # Load .npy
        image = np.load(
            file_path
        ).astype(np.float32)

        # ----------------------------------------------------
        # Expected input:
        #
        # H x W
        #
        # Convert to:
        #
        # 1 x H x W
        # ----------------------------------------------------

        if image.ndim == 2:

            image = torch.from_numpy(
                image
            ).unsqueeze(0)

        # ----------------------------------------------------
        # Already C x H x W
        # ----------------------------------------------------

        elif image.ndim == 3:

            image = torch.from_numpy(
                image
            )

        else:

            raise ValueError(
                f"Unsupported input shape "
                f"{image.shape} "
                f"for file: {file_path}"
            )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if not torch.isfinite(image).all():

            raise ValueError(
                f"NaN/Inf found in input: "
                f"{file_path}"
            )

        filename = os.path.basename(
            file_path
        )

        return image, filename


# ============================================================
# 5. FIND NPY FILES
# ============================================================

def find_npy_files(input_dir):

    paths = sorted(
        glob.glob(
            os.path.join(
                input_dir,
                "**",
                "*.npy"
            ),
            recursive=True
        )
    )

    return paths


# ============================================================
# 6. MAIN
# ============================================================

def main():

    # ========================================================
    # ARGUMENTS
    # ========================================================

    parser = argparse.ArgumentParser(
        description=(
            "KLA Hackathon 2026 "
            "Image Restoration Inference"
        )
    )

    parser.add_argument(
    "--input_dir",
    default="Data-public/Test_NoisyLR"
)

    parser.add_argument(
    "--output_dir",
    default="Restored_Test_Output"
)

    parser.add_argument(
    "--weights",
    default="weights/best_model.pth"
)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Inference batch size"
    )

    args = parser.parse_args()


    # ========================================================
    # DEVICE
    # ========================================================

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

        print("=" * 65)
        print("KLA HACKATHON 2026 - FINAL INFERENCE")
        print("=" * 65)

        print(
            "Device :",
            torch.cuda.get_device_name(0)
        )

    else:

        device = torch.device(
            "cpu"
        )

        print("=" * 65)
        print("WARNING: CUDA NOT AVAILABLE")
        print("Running inference on CPU")
        print("=" * 65)


    # ========================================================
    # CHECK INPUT DIRECTORY
    # ========================================================

    if not os.path.isdir(
        args.input_dir
    ):

        raise FileNotFoundError(
            f"\nInput directory not found:\n"
            f"{args.input_dir}"
        )


    # ========================================================
    # CHECK WEIGHTS
    # ========================================================

    if not os.path.isfile(
        args.weights
    ):

        raise FileNotFoundError(
            f"\nModel weights not found:\n"
            f"{args.weights}"
        )


    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )


    # ========================================================
    # FIND INPUT FILES
    # ========================================================

    npy_files = find_npy_files(
        args.input_dir
    )

    if len(npy_files) == 0:

        raise RuntimeError(
            f"\nNo .npy files found inside:\n"
            f"{args.input_dir}"
        )


    print()
    print(
        "Input directory :",
        args.input_dir
    )

    print(
        "Output directory:",
        args.output_dir
    )

    print(
        "Model weights   :",
        args.weights
    )

    print(
        "Input files     :",
        len(npy_files)
    )


    # ========================================================
    # CREATE MODEL
    #
    # EXACT SAME CONFIGURATION AS TRAINING
    # ========================================================

    model = NAFNetSR(
        in_channels=1,
        out_channels=1,
        dim=48,
        num_blocks=8,
        scale=2
    ).to(device)


    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    print()
    print("Loading trained checkpoint...")


    checkpoint = torch.load(
        args.weights,
        map_location=device,
        weights_only=True
    )


    # --------------------------------------------------------
    # Support:
    #
    # 1. plain state_dict
    #
    # 2. checkpoint containing model_state_dict
    # --------------------------------------------------------

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint


    # --------------------------------------------------------
    # STRICT loading
    #
    # IMPORTANT:
    # Do NOT use strict=False.
    #
    # We want to know if architecture is wrong.
    # --------------------------------------------------------

    model.load_state_dict(
        state_dict,
        strict=True
    )


    print(
        "Checkpoint loaded successfully."
    )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Model parameters: "
        f"{total_params:,}"
    )


  
    # DATA LOADER


    dataset = TestDataset(
        npy_files
    )


    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        )
    )


  
    # INFERENCE
    

    model.eval()

    print()
    print("Starting inference...")
    print("-" * 65)


    # Synchronize GPU before timing
    if device.type == "cuda":

        torch.cuda.synchronize()


    start_time = time.perf_counter()


    processed_images = 0


    with torch.inference_mode():

        for batch_idx, (
            noisy_batch,
            filenames
        ) in enumerate(loader):


           
            # Move input to GPU
           

            noisy_batch = noisy_batch.to(
                device,
                non_blocking=True
            )


         
            # MODEL FORWARD
            #
            # FP32 intentionally used here.
            # This keeps inference numerically consistent
            # with the trained checkpoint.
          

            restored_batch = model(
                noisy_batch
            )


          
            # Move output to CPU
           

            restored_batch = (
                restored_batch
                .float()
                .cpu()
                .numpy()
            )


            
            # SAVE OUTPUTS
          

            for i, filename in enumerate(
                filenames
            ):

                restored_image = (
                    restored_batch[i]
                    .squeeze()
                )
                # KLA evaluates the model output directly.
              

                output_path = os.path.join(
                    args.output_dir,
                    filename
                )


                np.save(
                    output_path,
                    restored_image.astype(
                        np.float32
                    )
                )


                processed_images += 1


            
            # Progress
           

            print(
                f"Processed "
                f"{processed_images}/"
                f"{len(npy_files)}",
                end="\r"
            )


   
    # GPU SYNCHRONIZATION
    

    if device.type == "cuda":

        torch.cuda.synchronize()


  
    # TIMING
  

    total_time = (
        time.perf_counter()
        - start_time
    )


    if processed_images > 0:

        avg_time = (
            total_time
            / processed_images
            * 1000
        )

        throughput = (
            processed_images
            / total_time
        )

    else:

        avg_time = 0.0
        throughput = 0.0


    
    # FINAL REPORT
   

    print()
    print()
    print("=" * 65)
    print("INFERENCE COMPLETE")
    print("=" * 65)

    print(
        f"Images processed : "
        f"{processed_images}"
    )

    print(
        f"Total time       : "
        f"{total_time:.3f} sec"
    )

    print(
        f"Average time     : "
        f"{avg_time:.3f} ms/image"
    )

    print(
        f"Throughput       : "
        f"{throughput:.2f} images/sec"
    )

    print(
        f"Outputs saved to : "
        f"{args.output_dir}"
    )

    print("=" * 65)

    print(
        "NO TRAINING WAS PERFORMED."
    )

    print(
        "MODEL WEIGHTS WERE NOT MODIFIED."
    )

    print("=" * 65)

# RUN

if __name__ == "__main__":

    main()