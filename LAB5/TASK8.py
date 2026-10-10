#Task 8: Tuning Watershed for Difficult Object SeparationThe following implementation modifies the distance-transform threshold to control how touching objects are separated.   Pythonimport cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load image and preprocess
image = cv2.imread('touching_coins.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Noise removal and sure background
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Distance transform
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

# Experiments with different distance-transform thresholds
multipliers = [0.1, 0.4, 0.6, 0.9]
results = []

plt.figure(figsize=(16, 4))

for i, mult in enumerate(multipliers):
    img_copy = image.copy()

    # Apply multiplier to determine sure foreground
    _, sure_fg = cv2.threshold(dist_transform, mult * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    # Label markers and count them
    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    cv2.watershed(img_copy, markers)
    img_copy[markers == -1] = [255, 0, 0] # Red boundaries

    # Plotting
    plt.subplot(1, 4, i+1)
    plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    plt.title(f'Multiplier: {mult}')
    plt.axis('off')

    results.append((mult, num_markers - 1)) # -1 to ignore background marker

plt.savefig('task8_watershed_tuning.png')
plt.show()
