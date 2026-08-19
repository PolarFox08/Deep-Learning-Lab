# Deep Learning Lab 4 — Comparative Study of Deep CNN Architectures Using Transfer Learning

**Course:** CS3807 -- Deep Learning Laboratory
**Degree & Branch:** B.Tech Artificial Intelligence & Data Science, Semester V
**AY:** 2026--27

## Overview

This lab studies the evolution of deep CNN architectures (LeNet-5, AlexNet, VGG16, GoogleNet, ResNet) and implements **transfer learning** using a pretrained **VGG16** model (ImageNet weights) with **TensorFlow/Keras**, fine-tuned for image classification on **CIFAR-10**.

## Contents

* `DL_Lab_4.ipynb` -- Main notebook: dataset loading/EDA, VGG16 transfer-learning setup (frozen base), model training, fine tuning (unfreezing the last convolution block), evaluation (accuracy, precision, recall, F1, confusion matrix, classification report), and a hyperparameter grid-search study.
* `DL_Lab4_Latex_Report.tex` -- LaTeX lab report.
* `DATASET_INFO.md` -- Details on the CIFAR-10 dataset and how to obtain it.
* `EXECUTION_INSTRUCTIONS.md` -- Steps to set up and run the notebook.
* `requirements.txt` -- Python dependencies.

## Tasks Covered

1. Load and prepare CIFAR-10 (normalize pixel values, one-hot encode labels, display sample images).
2. Set up transfer learning with a pretrained VGG16 base (ImageNet weights, classifier removed, base frozen, Global Average Pooling + Dense head added).
3. Train the classifier head with the VGG16 base frozen.
4. Fine tune by unfreezing the last convolution block and continuing training at a lower learning rate; compare accuracy before/after fine tuning.
5. Evaluate the trained model (accuracy, precision, recall, F1-score, confusion matrix, classification report).
6. Run a hyperparameter study (learning rate, batch size, epochs, optimizer, dense units, frozen layers).

See `EXECUTION_INSTRUCTIONS.md` to run the notebook and `DATASET_INFO.md` for dataset setup.
