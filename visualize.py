import os
import numpy as np
import matplotlib.pyplot as plt

noisy_dir = r"D:\KLA_Hackathon\Data-public\Test_NoisyLR\NoisyLR"
restored_dir = r"D:\KLA_Hackathon\Data-public\Restored_Test_Output"

# Pick first .npy file
sample_file = sorted([f for f in os.listdir(noisy_dir) if f.endswith('.npy')])[0]

noisy_img = np.load(os.path.join(noisy_dir, sample_file))
restored_img = np.load(os.path.join(restored_dir, sample_file))

print(f"Sample File: {sample_file}")
print(f"Noisy Input Shape: {noisy_img.shape}, Min: {noisy_img.min():.2f}, Max: {noisy_img.max():.2f}")
print(f"Restored Output Shape: {restored_img.shape}, Min: {restored_img.min():.2f}, Max: {restored_img.max():.2f}")

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title(f"Noisy Input ({noisy_img.shape[0]}x{noisy_img.shape[1]})")
plt.imshow(noisy_img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"NAFNet-SR Restored ({restored_img.shape[0]}x{restored_img.shape[1]})")
plt.imshow(restored_img, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()