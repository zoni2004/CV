import cv2
import matplotlib.pyplot as plt

image = cv2.imread('mechanical_part.jpg', cv2.IMREAD_GRAYSCALE)

# Test at least three different low/high threshold pairs
edges1 = cv2.Canny(image, 50, 100)  # Pair 1: Low-Low (captures everything)
edges2 = cv2.Canny(image, 50, 200)  # Pair 2: Low-High (High threshold increased)
edges3 = cv2.Canny(image, 150, 200) # Pair 3: High-High (Low threshold increased)

plt.figure(figsize=(15, 4))
plt.subplot(1, 4, 1), plt.imshow(image, cmap='gray'), plt.title('Original')
plt.subplot(1, 4, 2), plt.imshow(edges1, cmap='gray'), plt.title('Edge Result 1 (50, 100)')
plt.subplot(1, 4, 3), plt.imshow(edges2, cmap='gray'), plt.title('Edge Result 2 (50, 200)')
plt.subplot(1, 4, 4), plt.imshow(edges3, cmap='gray'), plt.title('Edge Result 3 (150, 200)')
plt.savefig('task5_output.png')
plt.show()
