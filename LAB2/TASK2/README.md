# CT–MRI Image Processing Pipeline

This project processes paired CT and MRI medical images using image enhancement, color mapping, overlay, and multimodal fusion techniques.

## Pipeline

```text
                    CT Image
                       │
                       ▼
              Histogram Equalization
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
      Color Map    Log Transform  Gamma Transform
          │          (γ = 0.5)      (γ = 0.5)
          ▼
       CT Heatmap


                    MRI Image
                       │
                       ▼
              Histogram Equalization
                       │
                       ▼
                  Color Mapping
                       │
                       ▼
                  MRI Heatmap


          CT Heatmap + MRI Heatmap
                       │
                       ▼
              Weighted Fusion
