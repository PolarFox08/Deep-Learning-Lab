# Deep Learning Lab 6 — End-to-End Study of RNN, LSTM and GRU for Sequence Learning and Video Understanding

**Course:** CS3807 -- Deep Learning Laboratory  
**Degree & Branch:** B.Tech Artificial Intelligence & Data Science, Semester V  
**AY:** 2026--27  

## Overview

This experiment develops an end-to-end understanding of recurrent sequence learning by implementing and comparing **Vanilla RNN**, **LSTM**, and **GRU** architectures. The lab explores Backpropagation Through Time (BPTT), vanishing/exploding gradient challenges, temporal sequence classification on 9-channel raw inertial sensor data, sequence length scaling, spatial-temporal video action recognition using a hybrid CNN-RNN pipeline, and sequence-to-sequence translation using an Encoder-Decoder framework.

The experiment evaluates models on two main tasks:
1. **Human Activity Recognition (UCI HAR):** Classifying 6 activities from 128-step 9-channel inertial sensor signals ($X \in \mathbb{R}^{N \times 128 \times 9}$).
2. **Video Action Recognition (UCF101 Subset):** Classifying 4 video action classes by extracting spatial feature sequences using a frozen ImageNet-pretrained **MobileNetV2** backbone ($X_{\text{video}} \in \mathbb{R}^{N \times 10 \times 1280}$) fed into LSTM or GRU recurrent layers.
3. **Synthetic Sequence Reversal:** Demonstrating an Encoder-Decoder LSTM architecture with Teacher Forcing for sequence-to-sequence transformation.

---

## Contents

| File / Folder | Description |
|---|---|
| `dl-lab-6.ipynb` | Main Jupyter notebook: raw inertial signal loading, preprocessing & 70/15/15 stratified split, BPTT numerical verification, RNN/LSTM/GRU model building & training, confusion matrix analysis, sequence length sweep $T \in \{32, 64, 128\}$, MobileNetV2 spatial feature extraction, CNN-LSTM/GRU video action classification, and synthetic seq2seq reversal model. |
| `Experiment_6.tex` | LaTeX lab report source containing mathematical derivations (RNN, LSTM, GRU equations, BPTT analysis), experimental setup, tables, plot inferences, and discussion questions. |
| `DL_Lab_6.pdf` | Compiled PDF report of Experiment 6. |
| `DATASET_INFO.md` | Details on the UCI HAR sensor dataset, UCF101 video subset, input tensor representations, class distributions, and splitting setup. |
| `EXECUTION_INSTRUCTIONS.md` | Step-by-step setup guide, dataset path configuration, and notebook execution instructions. |
| `requirements.txt` | Python package dependencies required to run the notebook. |
| `plots/` | Directory containing all 13 generated figures (sensor signals vs. time, training/validation loss & accuracy curves, confusion matrices, performance comparisons, sequence length analysis, video frames, and video training curves). |

---

## Tasks Covered

1. **Temporal Input Representation & Preprocessing:** Load 9 raw inertial signal channels from UCI HAR dataset, construct $(N, 128, 9)$ sample tensors across 6 activities, split into stratified 70% train / 15% validation / 15% test partitions, and normalize using training statistics.
2. **Temporal Data Visualization:** Plot sensor signal values vs. time steps across representative activities (WALKING, SITTING, LAYING) and channels (`body_acc_x`, `body_gyro_x`, `total_acc_x`) to analyze temporal motion patterns.
3. **Vanilla RNN & BPTT Analysis:** Implement Vanilla RNN, explain Backpropagation Through Time (BPTT), derive gradient update equations, analyze vanishing/exploding gradient problems, and perform manual vs. programmatic numerical verification.
4. **LSTM & GRU Architecture Implementation:** Build LSTM and GRU models, analyzing gating mechanics (forget, input, output gates in LSTM; reset, update gates in GRU) and cell state flow.
5. **Model Comparison & Convergence Study:** Train RNN, LSTM, and GRU under controlled settings (32 units, Adam $\text{lr}=10^{-3}$, 30 epochs) and compare convergence rate, final test accuracy/F1-score, model parameter count, and training duration.
6. **Confusion Matrix Analysis:** Evaluate per-class recall and misclassification patterns (e.g., RNN struggling with dynamic WALKING vs. LSTM/GRU accuracy on locomotion).
7. **Sequence Length Study:** Evaluate model performance across sequence lengths $T \in \{32, 64, 128\}$ to study long-term dependency retention.
8. **Spatial-Temporal Video Understanding (CNN + RNN):** Extract 1,280-dimensional frame feature vectors from UCF101 videos using frozen MobileNetV2, pass $(N, 10, 1280)$ sequences into CNN-LSTM and CNN-GRU classifiers, and evaluate action classification performance.
9. **Sequence-to-Sequence Learning:** Construct an Encoder-Decoder LSTM network with Teacher Forcing to learn integer sequence reversal, comparing token-level vs. sequence-level accuracy.

---

## Results Summary

| Model | Task / Dataset | Accuracy (%) | Macro F1 (%) | Parameters | Training Time (s) |
|---|---|:---:|:---:|:---:|:---:|
| **Vanilla RNN** | UCI HAR (Sequence) | 73.56 | 72.96 | 1,974 | 27.30 |
| **LSTM** | UCI HAR (Sequence) | 94.67 | 94.66 | 6,006 | 24.08 |
| **GRU** | UCI HAR (Sequence) | 95.11 | 95.10 | 4,758 | 20.88 |
| **CNN-LSTM** | UCF101 Video (4-class) | 100.00 | 100.00 | 168,677 | 12.45 |
| **CNN-GRU** | UCF101 Video (4-class) | 100.00 | 100.00 | 126,757 | 11.20 |
| **Seq2Seq LSTM** | Integer Reversal | 100.00 (Seq Acc) | 100.00 | 25,482 | 8.90 |

---

See `EXECUTION_INSTRUCTIONS.md` to run the code and `DATASET_INFO.md` for dataset setup.
