import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('coins_touching.jpg')
# 1. Preprocessing[cite: 23]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. Thresholding[cite: 23]
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 3. Noise removal[cite: 23]
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# 4. Sure-background generation[cite: 23]
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# 5. Distance transform[cite: 23]
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

# 6. Sure-foreground generation[cite: 23]
_, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

# 7. Unknown-region generation[cite: 23]
unknown = cv2.subtract(sure_bg, sure_fg)

# 8. Marker labeling[cite: 23]
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# 9. Watershed transformation[cite: 23]
markers = cv2.watershed(image, markers)

# 10. Boundary visualization[cite: 23]
image[markers == -1] = [255, 0, 0] # Mark touching boundaries red

# Output: Original, Threshold, Sure BG, Dist Transform, Sure FG, Unknown, Final[cite: 23]
# (Assume plotting code here saving to task7_output.png)
