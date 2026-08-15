import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# KLA HACKATHON 2026 - CLEAN VISUAL COMPARISON
# 3 IMAGE PAIRS
# ============================================================

NOISY_DIR = r"D:\KLA_Hackathon\Data-public\Test_NoisyLR\NoisyLR"
RESTORED_DIR = r"D:\KLA_Hackathon\Restored_Test_Output"

NUM_IMAGES = 3

# ============================================================
# FIND MATCHING FILES
# ============================================================

noisy_files = sorted([
    f for f in os.listdir(NOISY_DIR)
    if f.lower().endswith(".npy")
])

restored_files = set([
    f for f in os.listdir(RESTORED_DIR)
    if f.lower().endswith(".npy")
])

common_files = [
    f for f in noisy_files
    if f in restored_files
]

if len(common_files) < NUM_IMAGES:
    print("ERROR: 3 matching image pairs nahi mile.")
    raise SystemExit

selected = common_files[:NUM_IMAGES]

print("=" * 60)
print("KLA HACKATHON 2026 - VISUAL CHECK")
print("=" * 60)

print("Selected images:")
for f in selected:
    print(f)

# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    3,
    2,
    figsize=(12, 17)
)

fig.suptitle(
    "KLA HACKATHON 2026\nImage Restoration Results",
    fontsize=22,
    fontweight="bold",
    y=0.98
)

# ============================================================
# DISPLAY
# ============================================================

for i, filename in enumerate(selected):

    noisy_path = os.path.join(NOISY_DIR, filename)
    restored_path = os.path.join(RESTORED_DIR, filename)

    noisy = np.squeeze(np.load(noisy_path))
    restored = np.squeeze(np.load(restored_path))

    # --------------------------------------------------------
    # NOISY IMAGE
    # --------------------------------------------------------

    axes[i, 0].imshow(
        noisy,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    axes[i, 0].axis("off")

    # Text BELOW image
    axes[i, 0].text(
        0.5,
        -0.08,
        f"NOISY LR\n{noisy.shape[0]} × {noisy.shape[1]}",
        transform=axes[i, 0].transAxes,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold"
    )

    # --------------------------------------------------------
    # RESTORED IMAGE
    # --------------------------------------------------------

    axes[i, 1].imshow(
        restored,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    axes[i, 1].axis("off")

    # Text BELOW image
    axes[i, 1].text(
        0.5,
        -0.08,
        f"NAFNet-SR RESTORED\n{restored.shape[0]} × {restored.shape[1]}",
        transform=axes[i, 1].transAxes,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold"
    )
   
# ============================================================
# COLUMN LABELS
# ============================================================


fig.text(
    0.73,
    0.945,
    "RESTORED OUTPUT",
    ha="center",
    fontsize=16,
    fontweight="bold"
)

# ============================================================
# SPACING
# ============================================================

plt.subplots_adjust(
    left=0.08,
    right=0.92,
    top=0.88,
    bottom=0.05,
    wspace=0.25,
    hspace=0.75
)

# ============================================================
# SAVE
# ============================================================

RESULT_DIR = r"D:\KLA_Hackathon\results"
os.makedirs(RESULT_DIR, exist_ok=True)

save_path = os.path.join(
    RESULT_DIR,
    "visual_comparison_3_images.png"
)

plt.savefig(
    save_path,
    dpi=200,
    bbox_inches="tight"
)

print("\nVisualization saved:")
print(save_path)

plt.show()