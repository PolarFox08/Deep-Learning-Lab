# Deep Learning Lab 3 — Convolutional Neural Networks (CIFAR-10)

**Course:** CS3807 -- Deep Learning Laboratory
**Degree & Branch:** B.Tech Artificial Intelligence & Data Science, Semester V
**AY:** 2026--27

## Overview

This lab implements and studies Convolutional Neural Networks (CNNs) using **PyTorch**, covering the convolution operation, output-size calculations, feature map visualization, pooling, and end-to-end image classification on the **CIFAR-10** dataset.

## Contents

* `DL_Lab_3_pytorch.ipynb` -- Main notebook: dataset loading/EDA, convolution kernel-size and stride/padding experiments, feature map visualization, max vs. average pooling comparison, CNN construction and training, and full evaluation (accuracy, precision, recall, F1, confusion matrix).
* `DL_Lab3_Latex_Report.tex` -- LaTeX lab report.
* `DATASET_INFO.md` -- Details on the CIFAR-10 dataset and how to obtain it.
* `EXECUTION_INSTRUCTIONS.md` -- Steps to set up and run the notebook.
* `requirements.txt` -- Python dependencies.

## Tasks Covered

1. Load CIFAR-10 and explore the dataset (sample images, class distribution).
2. Compare convolution output sizes for $3\times3$, $5\times5$, and $7\times7$ kernels.
3. Study the effect of stride and padding on output dimensions.
4. Visualize feature maps after the first convolution layer.
5. Compare max pooling vs. average pooling.
6. Build and train a CNN: `Input -> Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool -> Flatten -> Dense -> Softmax`.
7. Evaluate the trained model (accuracy, precision, recall, F1-score, confusion matrix, classification report).

See `EXECUTION_INSTRUCTIONS.md` to run the notebook and `DATASET_INFO.md` for dataset setup.
