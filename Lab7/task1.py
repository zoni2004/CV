import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

img = cv2.imread("image.jpg")
if img is None:
    img = np.full((300, 400, 3), 255, np.uint8)
    cv2.rectangle(img, (100, 130), (300, 270), (60, 120, 220), -1)
    cv2.fillPoly(img, [np.array([[80, 130], [200, 40], [320, 130]])], (40, 40, 180))
    cv2.rectangle(img, (180, 190), (220, 270), (30, 70, 30), -1)
    cv2.putText(img, "FAST", (150, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

h, w = img.shape[:2]


def translation(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], float)


def rotation(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def scaling(sx, sy):
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], float)


def shearing(kx, ky):
    return np.array([[1, kx, 0], [ky, 1, 0], [0, 0, 1]], float)


def apply(image, matrix):
    return cv2.warpAffine(image, matrix[:2], (w, h), borderValue=(255, 255, 255))


def show(images, titles, name):
    plt.figure(figsize=(4 * len(images), 4))
    for i, (pic, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(cv2.cvtColor(pic, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("results/" + name + ".png", dpi=120)
    plt.show()


centre = translation(w / 2, h / 2)
back = translation(-w / 2, -h / 2)

pictures = [
    img,
    apply(img, translation(80, 40)),
    apply(img, centre @ rotation(30) @ back),
    apply(img, centre @ scaling(0.6, 0.6) @ back),
    apply(img, centre @ shearing(0.3, 0) @ back),
]
show(pictures, ["Original", "Translation", "Rotation", "Scaling", "Shear"], "q1a_2d")

cube = np.array([[x, y, z, 1] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
pairs = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.sum(cube[i, :3] != cube[j, :3]) == 1]


def rotate3d(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    m = np.eye(4)
    i, j = [(1, 2), (0, 2), (0, 1)][axis]
    m[i, i], m[i, j], m[j, i], m[j, j] = c, -s, s, c
    return m


def move3d(x, y, z):
    m = np.eye(4)
    m[:3, 3] = [x, y, z]
    return m


M = move3d(3, 2, 1) @ rotate3d(2, 40) @ rotate3d(1, 30) @ rotate3d(0, 20)
moved = (M @ cube.T).T

ax = plt.figure(figsize=(7, 6)).add_subplot(111, projection="3d")
for data, color, name in [(cube, "blue", "Original"), (moved, "red", "Rotated and moved")]:
    for i, j in pairs:
        ax.plot([data[i, 0], data[j, 0]], [data[i, 1], data[j, 1]], [data[i, 2], data[j, 2]], color=color)
    ax.plot([], [], color=color, label=name)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.savefig("results/q1b_3d.png", dpi=120)
plt.show()

point = np.array([4, 3])
homogeneous = np.append(point, 1)
print("2D point:", point)
print("Homogeneous point:", homogeneous)
print("Translated by (2, 5):", translation(2, 5) @ homogeneous)
print("Scaled by 2 and rotated 90 degrees:", (rotation(90) @ scaling(2, 2) @ homogeneous).round(2))
print("The point (8, 6, 2) is the same as (4, 3, 1):", (np.array([8, 6, 2]) / 2))
print("The extra coordinate lets translation be written as a matrix, lets all transforms be combined by")
print("multiplying matrices, and makes perspective possible by dividing by the last value.")

grid = np.full((300, 300, 3), 255, np.uint8)
for k in range(0, 301, 30):
    cv2.line(grid, (k, 0), (k, 300), (0, 0, 0), 1)
    cv2.line(grid, (0, k), (300, k), (0, 0, 0), 1)
before = np.float32([[0, 0], [299, 0], [0, 299]])
after = np.float32([[30, 50], [250, 20], [70, 280]])
A = cv2.getAffineTransform(before, after)
affine_result = cv2.warpAffine(grid, A, (300, 300), borderValue=(255, 255, 255))
show([grid, affine_result], ["Original grid", "After affine transform"], "q1d_affine")

line_a = np.array([0, 0, 1.0])
line_b = np.array([100, 50, 1.0])
line_c = np.array([200, 100, 1.0])
A3 = np.vstack([A, [0, 0, 1]])
a, b, c = A3 @ line_a, A3 @ line_b, A3 @ line_c
print("Determinant of the three transformed points (0 means still on one line):", round(np.linalg.det(np.array([a, b, c])), 6))
print("Straight lines stay straight and parallel lines stay parallel after an affine transform.")

book = np.full((300, 400, 3), 255, np.uint8)
cv2.rectangle(book, (5, 5), (394, 294), (0, 0, 0), 5)
cv2.putText(book, "BOOK", (110, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, (150, 40, 20), 4)
flat_corners = np.float32([[0, 0], [399, 0], [399, 299], [0, 299]])
slanted_corners = np.float32([[70, 50], [360, 100], [320, 280], [30, 230]])
slant = cv2.warpPerspective(book, cv2.getPerspectiveTransform(flat_corners, slanted_corners), (400, 300),
                            borderValue=(200, 200, 200))
H, mask = cv2.findHomography(slanted_corners, flat_corners)
print("Homography:\n", H.round(4))
fixed = cv2.warpPerspective(slant, H, (400, 300))
show([book, slant, fixed], ["Flat book", "Slanted book", "Corrected"], "q1e_homography")
