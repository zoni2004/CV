import cv2
import matplotlib.pyplot as plt
import numpy as np

CT = cv2.imread('CT.png', cv2.IMREAD_GRAYSCALE)
MRI = cv2.imread('slice_037.png', cv2.IMREAD_GRAYSCALE)

if CT.shape != MRI.shape:
    MRI = cv2.resize(MRI, (CT.shape[1], CT.shape[0]))

CT_equalized = cv2.equalizeHist(CT)
MRI_equalized = cv2.equalizeHist(MRI)

CT_heatmap = cv2.applyColorMap(CT_equalized, cv2.COLORMAP_BONE)
MRI_heatmap = cv2.applyColorMap(MRI_equalized, cv2.COLORMAP_HOT)

fused = cv2.addWeighted(CT_heatmap, 0.7, MRI_heatmap, 0.3, 0)

image_float = fused.astype(np.float32)
c = 255/(np.log(1+np.max(image_float)))
log_transformed = c * np.log(1+image_float)
log_transformed = log_transformed.astype(np.uint8)

gamma_corrected = np.array(255*(fused / 255) ** 0.5, dtype = 'uint8')

plt.figure(figsize=(14, 8))

plt.subplot(2, 4, 1)
plt.imshow(CT, cmap='gray')
plt.title('Original CT')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(CT_equalized, cmap='gray')
plt.title('Equalized CT')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(cv2.cvtColor(CT_heatmap, cv2.COLOR_BGR2RGB))
plt.title('CT Heatmap')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(cv2.cvtColor(MRI_heatmap, cv2.COLOR_BGR2RGB))
plt.title('MRI Heatmap')
plt.axis('off')

plt.subplot(2, 4, 5)
plt.imshow(cv2.cvtColor(fused, cv2.COLOR_BGR2RGB))
plt.title('Weighted Fusion')
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(log_transformed, cmap='gray')
plt.title('Log Transformation')
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(gamma_corrected, cmap='gray')
plt.title('Gamma Transformation (γ = 0.5)')
plt.axis('off')

plt.tight_layout()
plt.show()
