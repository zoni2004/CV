import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image as grayscale (CV_8UC1)
image = cv2.imread('COVID.png', cv2.IMREAD_GRAYSCALE)

# Equalize the histogram of the grayscale image
equalized_image = cv2.equalizeHist(image)

# Plot the results
plt.figure(figsize=(10, 5))

plt.subplot(4, 2, 1)
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.title('Original Image')

plt.subplot(4, 2, 2)
plt.imshow(equalized_image, cmap='gray')
plt.axis('off')
plt.title('Equalized Image')

false_color_heatmap = cv2.applyColorMap(equalized_image, cv2.COLORMAP_JET)
plt.subplot(4 ,2 ,3)
plt.imshow(false_color_heatmap, cmap='gray')
plt.title('False Color Heatmap')
plt.axis('off')

#COLOR BALANCING
#convert to float for mathematical operations
img_float = false_color_heatmap.astype('float32')
#find mean of each channel
b_mean = img_float[:, :, 0].mean()
g_mean = img_float[:, :, 1].mean()
r_mean = img_float[:, :, 2].mean()
#calculate total mean
mean = (b_mean+g_mean+r_mean)/3
#balance the channels
img_float[:, :, 0] *= mean / b_mean
img_float[:, :, 1] *= mean / g_mean
img_float[:, :, 2] *= mean / r_mean
#normalize
color_balanced = cv2.normalize(img_float, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

plt.subplot(4 ,2 ,4)
plt.imshow(color_balanced)
plt.title('Color Balanced')
plt.axis('off')

ret, threshold_image = cv2.threshold(equalized_image, 100, 255, cv2.THRESH_BINARY)
plt.subplot(4 ,2 ,5)
plt.imshow(threshold_image, cmap='gray')
plt.title('Threshold Image')
plt.axis('off')

image_float = image.astype(np.float32)
c = 255/(np.log(1+np.max(image_float)))
log_transformed = c * np.log(1+image_float)
log_transformed = log_transformed.astype(np.uint8)

plt.subplot(4 ,2 ,6)
plt.imshow(log_transformed, cmap='gray')
plt.title('Log Transformed')
plt.axis('off')

# Apply gamma correction.
gamma_corrected = np.array(255*(image / 255) ** 0.5, dtype = 'uint8')
# Save edited images.
cv2.imwrite('gamma_transformed'+str(0.5)+'.jpg', gamma_corrected)

plt.subplot(4 ,2 ,7)
plt.imshow(gamma_corrected, cmap='gray')
plt.title('Gamma Transformed')
plt.axis('off')
plt.show()
