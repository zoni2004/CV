import cv2
import matplotlib.pyplot as plt

# 1. Convert the image to grayscale[cite: 19]
image = cv2.imread('coins.jpg', cv2.IMREAD_GRAYSCALE)

# 3. Apply Otsu's thresholding[cite: 19]
otsu_thresh_val, otsu_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 5. Repeat the experiment after changing contrast[cite: 19]
# Increasing contrast via histogram equalization
image_eq = cv2.equalizeHist(image)
otsu_thresh_eq, otsu_mask_eq = cv2.threshold(image_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print(f"Original Otsu Threshold: {otsu_thresh_val}") # 4. Record threshold[cite: 19]
print(f"High-Contrast Otsu Threshold: {otsu_thresh_eq}") # 6. Compare[cite: 19]

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Original')
plt.subplot(1, 3, 2), plt.hist(image.ravel(), 256, [0, 256]), plt.title('Histogram') # 2. Generate histogram[cite: 19]
plt.subplot(1, 3, 3), plt.imshow(otsu_mask, cmap='gray'), plt.title('Otsu Binary Mask')
plt.savefig('task3_output.png')
plt.show()
