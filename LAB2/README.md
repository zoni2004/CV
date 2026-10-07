# Computer Vision Lab 2

This repository contains the solutions for Computer Vision Lab 2, focusing on fundamental image and video processing techniques applied to medical imaging.

## Tasks

### TASK 1 — X-ray Image Enhancement

Processes a chest X-ray using techniques such as:

- Histogram Equalization
- False-Color Mapping
- Color Balancing
- Thresholding
- Logarithmic Transformation
- Power-Law (Gamma) Transformation

### TASK 2 — CT–MRI Image Fusion

Processes CT and MRI images and performs:

- Histogram Equalization
- Color Mapping
- Multimodal Weighted Fusion
- Logarithmic Transformation
- Power-Law (Gamma) Transformation

### TASK 3 — Real-Time Ultrasound Processing

Processes an ultrasound echocardiography video frame-by-frame using:

- Video Capture
- Grayscale Conversion
- Histogram Equalization
- JET Color Mapping
- Color Balancing
- Logarithmic Transformation
- Power-Law (Gamma) Transformation
- Side-by-side Raw vs Enhanced Display

## Repository Structure

```text
LAB2/
│
├── TASK1/
│   ├── data/
│   ├── output/
│   ├── README.md
│   └── xray_enhacement.py
│
├── TASK2/
│   ├── data/
│   ├── output/
│   ├── README.md
│   └── modal_fusion.py
│
├── TASK3/
│   ├── data/
│   ├── output/
│   ├── README.md
│   └── realtime_echo.py
│
├── requirements.txt
└── README.md
