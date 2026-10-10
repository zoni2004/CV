import cv2
import matplotlib.pyplot as plt

# 1. Load the image in grayscale
image = cv2.imread('document.jpg', cv2.IMREAD_GRAYSCALE)

# 2. Apply global thresholding using at least three different threshold values[cite: 17]
_, t1 = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY)  # Low
_, t2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY) # Medium
_, t3 = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY) # High

# 4. Apply adaptive thresholding to the same image[cite: 17]
# Using Gaussian with a 15x15 neighborhood and a constant C of 5
adaptive = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)

# Required Output: Create a comparison containing all results
titles = ['Original', 'Global T1 (80)', 'Global T2 (127)', 'Global T3 (170)', 'Adaptive']
images = [image, t1, t2, t3, adaptive]

plt.figure(figsize=(15, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.savefig('task1_output.png') # Save final output[cite: 17]
plt.show()
