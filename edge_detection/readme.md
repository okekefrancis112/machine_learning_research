# Lane Detection with Canny Edge Detection

This project is a Python-based implementation of a lane detection pipeline using a simplified Canny edge detection algorithm. It processes an input image (`lane.png`) to detect lane lines through smoothing, gradient computation, non-maximum suppression (NMS), and thresholding. The application uses OpenCV and NumPy for image processing and visualization.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Media](#media)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Image Smoothing**: Applies a Gaussian-like kernel to reduce noise in the input image.
- **Gradient Computation**: Calculates gradient magnitude and direction using Sobel filters to identify edges.
- **Non-Maximum Suppression (NMS)**: Suppresses non-maximum edge pixels based on gradient direction for sharper edges.
- **Thresholding**: Applies a high threshold to retain strong edge pixels, highlighting potential lane lines.
- **Visualization**: Displays intermediate and final results using OpenCV windows.

## Installation

### Prerequisites
- Python 3.7–3.11 (Python 3.13 may cause issues with some dependencies).
- A compatible operating system (Windows, macOS, or Linux).
- An input image named `lane.png` in the project directory.
- `pip` or `conda` for package management.

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/lane-detection.git
   cd lane-detection