import cv2
import numpy as np
from google.colab.patches import cv2_imshow

video = cv2.VideoCapture('echo.mp4')

while True:
    ret, frame_original = video.read()

    if not ret:
        break

    frame = cv2.cvtColor(frame_original, cv2.COLOR_BGR2GRAY)

    frame_equalized = cv2.equalizeHist(frame)

    frame_colored_heatmap = cv2.applyColorMap(frame_equalized, cv2.COLORMAP_JET)

    img_float = frame_colored_heatmap.astype(np.float32)

    b_mean = img_float[:, :, 0].mean()
    g_mean = img_float[:, :, 1].mean()
    r_mean = img_float[:, :, 2].mean()

    mean = (b_mean + g_mean + r_mean) / 3

    img_float[:, :, 0] *= mean / b_mean
    img_float[:, :, 1] *= mean / g_mean
    img_float[:, :, 2] *= mean / r_mean

    frame_color_balanced = np.clip(img_float, 0, 255).astype(np.uint8)

    frame_float = frame.astype(np.float32)

    c = 255 / np.log(1 + np.max(frame_float))
    frame_log_transformed = c * np.log(1 + frame_float)

    frame_log_transformed = np.clip(frame_log_transformed, 0, 255).astype(np.uint8)

    gamma = 0.5

    frame_float = frame.astype(np.float32) / 255.0

    gamma_corrected = np.power(frame_float,gamma)

    gamma_corrected = (gamma_corrected * 255).astype(np.uint8)

    combined = cv2.hconcat([frame_original,frame_color_balanced])

    cv2_imshow(combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
