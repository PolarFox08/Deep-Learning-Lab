# Dataset Information — Deep Learning Lab 6

This lab utilizes three distinct datasets to evaluate sequence modeling, spatial-temporal feature extraction, and sequence-to-sequence translation.

---

## 1. Primary Dataset: UCI Human Activity Recognition (HAR) Using Smartphones

### Overview
* **Dataset Name:** UCI Human Activity Recognition Using Smartphones
* **Source:** UCI Machine Learning Repository ([Dataset #240](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)) / Kaggle (`uvaiszufar/uci-har`)
* **Sensors & Channels:** 9 raw inertial signal channels:
  * 3-axis Body Acceleration (`body_acc_x`, `body_acc_y`, `body_acc_z`)
  * 3-axis Body Gyroscope (`body_gyro_x`, `body_gyro_y`, `body_gyro_z`)
  * 3-axis Total Acceleration (`total_acc_x`, `total_acc_y`, `total_acc_z`)
* **Window Size:** 128 time steps per sample window (sampled at 50 Hz, 2.56-second duration with 50% overlap).
* **Input Tensor Shape:** $X \in \mathbb{R}^{N \times 128 \times 9}$ (Samples $\times$ Sequence Length $\times$ Features).
* **Target Classes (6 Activities):**
  1. `WALKING`
  2. `WALKING_UPSTAIRS`
  3. `WALKING_DOWNSTAIRS`
  4. `SITTING`
  5. `STANDING`
  6. `LAYING`

### Experimental Subset & Split Protocol
* **Subset Selection:** Stratified sampling of 3,000 windows (500 samples per class) drawn from the combined raw signal pool (10,299 total windows).
* **Data Splits (70 / 15 / 15 Protocol):**

| Split | Percentage | Sample Count | Purpose |
|---|---|---|---|
| **Train Set** | 70% | 2,100 | Model training & gradient updates |
| **Validation Set** | 15% | 450 | Hyperparameter tuning & epoch-wise validation |
| **Test Set (Held-Out)** | 15% | 450 | Unseen final evaluation touched once per model |

* **Normalization:** Input features are normalized using per-channel mean and standard deviation calculated exclusively from the 2,100-sample training set.

---

## 2. Secondary Dataset: UCF101 Action Recognition (Video Subset)

### Overview
* **Dataset Name:** UCF101 Action Recognition Data (Subset)
* **Source:** Kaggle (`matthewjansen/ucf101-action-recognition`)
* **Video Subset:** 514 video clips across 4 action classes:
  * `Basketball`
  * `Biking`
  * `WalkingWithDog`
  * `TennisSwing`
* **Frame Sampling & Resolution:** 10 uniform frames per video clip, resized to $224 \times 224 \times 3$ RGB.

### Spatial Feature Extraction (CNN Backbone)
* **Pretrained Model:** Frozen MobileNetV2 (ImageNet weights, top layer excluded, global average pooling).
* **Frame Feature Dimension:** 1,280 features per frame.
* **Extracted Sequence Tensor:** $X_{\text{video}} \in \mathbb{R}^{N_{\text{videos}} \times 10 \times 1280}$.

---

## 3. Synthetic Dataset: Sequence-to-Sequence Integer Reversal

### Overview
* **Task:** Reversing a 4-token integer sequence from a vocabulary of digits 0–9 (e.g., $[1, 4, 7, 2] \to [2, 7, 4, 1]$).
* **Dataset Size:** 10,000 synthetic sequence pairs (8,000 train, 2,000 validation).
* **Input / Target Tensor Shape:** $(N, 4)$ integer matrices.
* **Framework:** Encoder-Decoder LSTM architecture with Teacher Forcing during training.
