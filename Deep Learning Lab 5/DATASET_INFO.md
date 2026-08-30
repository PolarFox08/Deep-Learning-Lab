# Dataset Information — Oxford-IIIT Pet Dataset

## Overview

* **Dataset Name:** Oxford-IIIT Pet Dataset
* **Source:** Kaggle (`tanlikesmath/the-oxfordiiit-pet-dataset`) / Visual Geometry Group, University of Oxford ([https://www.robots.ox.ac.uk/~vgg/data/pets/](https://www.robots.ox.ac.uk/~vgg/data/pets/))
* **Total Images:** 7,389 RGB images
* **Number of Classes:** 37 breeds (25 dog breeds, 12 cat breeds)
* **Species Breakdown:** 4,989 dog images, 2,400 cat images (~191–200 images per breed)
* **Input Resolution:** Resized to $224 \times 224 \times 3$ (RGB)
* **Preprocessing:** MobileNetV2 scaling via `tensorflow.keras.applications.mobilenet_v2.preprocess_input` (maps pixel values from $[0, 255]$ to $[-1, 1]$)

## Label Parsing Convention

The dataset consists of flat JPEG image files named according to the pattern `<breed>_<number>.jpg` (e.g., `saint_bernard_70.jpg`, `Siamese_67.jpg`). Labels are extracted dynamically from filenames:
1. Everything preceding the trailing `_<number>` constitutes the breed name.
2. A breed name starting with an **uppercase** letter represents a **cat** breed (e.g., `Abyssinian`, `Siamese`, `Bengal`, `Ragdoll`).
3. A breed name starting with a **lowercase** letter represents a **dog** breed (e.g., `beagle`, `boxer`, `pug`, `saint_bernard`).

*Note:* To prevent file duplication, the nested directory `images/images/` is used as the single source of truth, bypassing stray/duplicate outer files (such as `boxer_16.jpg`).

## Dataset Splits

The dataset is partitioned using class-stratified splits with a fixed random seed (`seed = 42`):

| Split | Percentage | Image Count | Usage |
|---|---|---|---|
| **Train Set** | 64% | 4,728 | Base training set for transfer learning & fine-tuning |
| **Validation Set** | 16% | 1,183 | Hyperparameter monitoring during transfer learning |
| **Test Set (Held-Out)** | 20% | 1,478 | Touched **exactly once** in Section 13 for final model evaluation |
| **Trend Sub-Train** | 60% of Train | 2,836 | Controlled from-scratch trend experiments (Sections 5–9) |
| **Trend Sub-Val** | 60% of Val | 709 | Validation monitoring for trend experiments (Sections 5–9) |
| **CV Training Pool** | Train + Val | 5,911 | Used for genuine 5-fold stratified cross-validation (Section 12) |

## Class Distribution

The dataset is balanced across breeds, averaging ~200 images per breed:
* Minimum images per breed: 191
* Maximum images per breed: 200
* Mean images per breed: 199.7

## Pretrained Model Weights

For transfer learning experiments (Sections 10–13), **MobileNetV2 ImageNet weights** are downloaded automatically via `tensorflow.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))` on the first run. An active internet connection is required for initial weight download.
