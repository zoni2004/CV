import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load a color image
image = cv2.imread('wildlife.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2. Represent the image as a set of pixel feature vectors[cite: 25]
pixel_values = image_rgb.reshape((-1, 3))

# 3. Convert the pixel data to the appropriate numeric type[cite: 25]
pixel_values = np.float32(pixel_values)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

k_results = []
# Run the experiment for K=2, 4 and 6[cite: 25]
for K in [2, 4, 6]:
    # 4. Perform K-Means clustering[cite: 25]
    _, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    # 5. Reconstruct the segmented image[cite: 25]
    segmented_image = centers[labels.flatten()].reshape(image_rgb.shape)
    k_results.append((K, segmented_image))

# (Plotting code omitted, saving to task9_output.png)
