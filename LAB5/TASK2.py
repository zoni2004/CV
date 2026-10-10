import cv2
import matplotlib.pyplot as plt

image = cv2.imread('document.jpg', cv2.IMREAD_GRAYSCALE)

# Testing at least three different blockSize values and C values[cite: 18]
# Both Mean-based and Gaussian-based[cite: 18]
results = [
    ("Mean, Block=5, C=2", cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)),
    ("Mean, Block=35, C=2", cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 2)),
    ("Mean, Block=15, C=15", cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 15)),
    ("Gaussian, Block=5, C=2", cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)),
    ("Gaussian, Block=35, C=2", cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 2)),
    ("Gaussian, Block=15, C=5", cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5))
]

plt.figure(figsize=(18, 6))
plt.subplot(2, 4, 1), plt.imshow(image, cmap='gray'), plt.title("Original"), plt.axis('off')
for i, (title, img) in enumerate(results):
    plt.subplot(2, 4, i+2), plt.imshow(img, cmap='gray'), plt.title(title), plt.axis('off')
plt.savefig('task2_output.png')
plt.show()
