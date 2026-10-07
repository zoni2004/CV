# CT–MRI Image Processing Pipeline

This project processes paired CT and MRI medical images using image enhancement, color mapping, overlay, and multimodal fusion techniques.

## Pipeline

```text
              MRI Image & CT image
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
          ┌────────────┼────────────┐
          │                         │
          ▼                         ▼
     Log Transform              Gamma Transform
                                 (γ = 0.5)
