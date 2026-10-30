# Sobel vs. Canny Edge Detection Benchmark

This repository contains a Python script designed to quantitatively evaluate and compare the performance of the **Sobel** and **Canny** edge detection algorithms. 

Instead of relying solely on visual inspection, this script calculates actual machine learning metrics (Precision, Recall, F1-Score, and Accuracy) by comparing the algorithms' outputs against ground-truth edges extracted from image masks.

## Features
* **Automated Benchmarking:** Processes a batch of images and calculates average performance metrics.
* **Tolerant Evaluation:** Uses a 1-pixel dilation technique on the ground truth to prevent overly harsh penalties for edges that are visually accurate but shifted by a single pixel.
* **Image Preprocessing:** Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) and Gaussian blurring to optimize images before edge detection.
* **Visualization:** Automatically displays a side-by-side Matplotlib comparison of the original image, mask, ground truth, Sobel output, and Canny output for the first 3 images processed.

## Prerequisites

You will need Python 3.x installed along with the following libraries:

```bash
pip install opencv-python numpy matplotlib
```

## Directory Structure

The script expects a specific folder structure to run correctly. By default, it looks for a `date` folder in the same directory as the script, containing an `imgs` folder and a `masks` folder. 

Mask files should have the exact same name as their corresponding image file, but with `_mask` appended before the extension (e.g., `image1.png` -> `image1_mask.png`).

```text
project_folder/
│
├── benchmark.py         # The main script
└── date/
    ├── imgs/            # Place your original images here
    │   ├── 001.png
    │   └── 002.png
    └── masks/           # Place your ground-truth silhouettes/masks here
        ├── 001_mask.png
        └── 002_mask.png
```

## How to Run

Simply execute the Python script from your terminal:

```bash
python benchmark.py
```

*Note: The script will pause when it displays the visual plots for the first three images. You must close the plot window for the script to continue processing the rest of the dataset.*

## How it Works

1. **Preprocessing:** The raw image is converted to grayscale, contrast-enhanced using CLAHE, and blurred slightly to reduce noise.
2. **Ground Truth Generation:** The script loads the corresponding mask, binarizes it, and extracts its outline to serve as the "perfect answer" (`gt_edges`).
3. **Edge Detection:** 
   * **Sobel:** Calculates X and Y gradients, combines their magnitude, and uses Otsu's thresholding to create a binary edge map.
   * **Canny:** Applies the standard Canny algorithm with fixed thresholds (30, 100).
4. **Evaluation:** The script compares the predictions to the ground truth. It applies a 3x3 dilation to the ground truth to create a 1-pixel "forgiveness buffer" for predicted edges.
5. **Scoring:** Calculates Precision (accuracy of detected edges), Recall (percentage of real edges found), F1-Score, and overall Accuracy.

## Results

After processing all images in the folder, the script will output a final scoreboard to the console, showing the average performance of both algorithms across your entire dataset.