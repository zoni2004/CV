import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the image in color
image = cv2.imread('yellow_car.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2. Convert it to HSV[cite: 20]
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Mask 1: Too restrictive[cite: 20]
restrictive_lower = np.array([25, 150, 150])
restrictive_upper = np.array([35, 255, 255])
mask_restrictive = cv2.inRange(hsv_image, restrictive_lower, restrictive_upper)

# Mask 2: Final mask that captures the object more completely[cite: 20]
# 3. Define a lower and upper color range[cite: 20]
final_lower = np.array([15, 70, 70])
final_upper = np.array([45, 255, 255])
# 4. Generate a binary mask[cite: 20]
mask_final = cv2.inRange(hsv_image, final_lower, final_upper)

# 6. Extract the selected color from the original image[cite: 20]
extracted_object = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_final)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1), plt.imshow(image_rgb), plt.title('Original')
plt.subplot(1, 3, 2), plt.imshow(mask_restrictive, cmap='gray'), plt.title('Restrictive Mask')
plt.subplot(1, 3, 3), plt.imshow(extracted_object), plt.title('Extracted Object')
plt.savefig('task4_output.png')
plt.show()
