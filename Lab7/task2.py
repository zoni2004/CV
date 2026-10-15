import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_perspective_transform(image, source_corners, width, height):
    destination = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    H, mask = cv2.findHomography(np.float32(source_corners), destination)
    return cv2.warpPerspective(image, H, (width, height)), H


def sort_corners(points):
    points = points.reshape(4, 2).astype("float32")
    total = points.sum(axis=1)
    difference = np.diff(points, axis=1).ravel()
    return np.float32([points[np.argmin(total)], points[np.argmin(difference)],
                       points[np.argmax(total)], points[np.argmax(difference)]])


def find_document_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            return sort_corners(approx)
    return None


def sample_image():
    page = np.full((400, 300, 3), 255, np.uint8)
    cv2.rectangle(page, (0, 0), (299, 399), (0, 0, 0), 6)
    for i, text in enumerate(["COMPUTER", "VISION", "LAB 07", "FAST NUCES"]):
        cv2.putText(page, text, (30, 90 + i * 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (160, 40, 20), 2)
    flat = np.float32([[0, 0], [299, 0], [299, 399], [0, 399]])
    tilted = np.float32([[120, 60], [430, 20], [480, 470], [60, 400]])
    matrix = cv2.getPerspectiveTransform(flat, tilted)
    return cv2.warpPerspective(page, matrix, (560, 520), borderValue=(90, 90, 90))


image = cv2.imread(sys.argv[1]) if len(sys.argv) > 1 else sample_image()
corners = find_document_corners(image)
if corners is None:
    print("Could not find a four sided shape. Edit the corners below by hand.")
    corners = np.float32([[120, 60], [430, 20], [480, 470], [60, 400]])

top_left, top_right, bottom_right, bottom_left = corners
width = int(max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left)))
height = int(max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right)))

result, H = apply_perspective_transform(image, corners, width, height)
print("Corners found:\n", corners)
print("Homography:\n", H.round(4))

marked = image.copy()
for x, y in corners:
    cv2.circle(marked, (int(x), int(y)), 8, (0, 0, 255), -1)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB))
plt.title("Detected corners")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Head on view")
plt.axis("off")
plt.tight_layout()
plt.savefig("q2_result.png", dpi=120)
plt.show()
