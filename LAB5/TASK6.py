import cv2
import numpy as np
import matplotlib.pyplot as plt

def region_growing(image, seed, threshold):
    mask = np.zeros_like(image, dtype=np.uint8)
    stack = [seed]
    seed_intensity = image[seed]

    while stack:
        x, y = stack.pop()
        if x < 0 or x >= image.shape[0] or y < 0 or y >= image.shape[1]: continue
        if mask[x, y] == 0:
            if abs(int(image[x, y]) - int(seed_intensity)) < threshold:
                mask[x, y] = 255
                stack.extend([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])
    return mask

# 1. Load a grayscale image
image = cv2.imread('brain_mri.jpg', cv2.IMREAD_GRAYSCALE)

# 2. Select seed points[cite: 22]
seed1 = (100, 100) # Assuming this points to white matter
seed2 = (150, 150) # Assuming this points to gray matter/tumor

# Experiment with three thresholds[cite: 22]
mask_t10 = region_growing(image, seed1, threshold=10)
mask_t30 = region_growing(image, seed1, threshold=30)
mask_t50 = region_growing(image, seed1, threshold=50)

# Experiment with different seed[cite: 22]
mask_seed2 = region_growing(image, seed2, threshold=30)

# (Plotting code omitted for brevity; assume subplot layout saving to task6_output.png)
