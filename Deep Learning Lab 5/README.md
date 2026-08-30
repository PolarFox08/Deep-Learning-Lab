# Deep Learning Lab 5 — Comprehensive Study of CNN Training, Regularization, Optimization, Hyperparameter Tuning, Transfer Learning and Cross-Validation

**Course:** CS3807 -- Deep Learning Laboratory  
**Degree & Branch:** B.Tech Artificial Intelligence & Data Science, Semester V  
**AY:** 2026--27  

## Overview

This experiment systematically investigates the impact of weight initialization, regularization strategies, optimization algorithms, CNN hyperparameters, transfer learning, fine-tuning, and $K$-fold cross-validation on image classification performance. 

The lab utilizes the lightweight **MobileNetV2** architecture and the **Oxford-IIIT Pet Dataset** (37 cat & dog breeds). Students analyze training/validation curves across individual design choices trained from scratch on a stratified subset, perform transfer learning with an ImageNet-pretrained backbone, and select a robust final configuration using 5-fold cross-validation evaluated on an independent held-out test set.

## Contents

| File / Folder | Description |
|---|---|
| `dl-lab-5.ipynb` | Main Jupyter notebook: dataset loading & label parsing, weight initialization comparison, regularization & Batch Normalization studies, optimizer sweep, hyperparameter grid search, MobileNetV2 transfer learning & fine-tuning, 5-fold cross-validation, and final model evaluation. |
| `Experiment_5.tex` | LaTeX lab report source containing theoretical background, mathematical derivations, experimental setup, results tables, plot inferences, and discussion questions. |
| `DATASET_INFO.md` | Details on the Oxford-IIIT Pet dataset, breed parsing rules, class distributions, and data splitting setup. |
| `EXECUTION_INSTRUCTIONS.md` | Step-by-step instructions to set up the environment, download the dataset, and execute the notebook. |
| `requirements.txt` | Python package dependencies required to run the notebook. |
| `plots/` | Directory containing all 15 generated figures (training/validation loss and accuracy curves, confusion matrix, and misclassified samples). |

## Tasks Covered

1. **Dataset Exploration & Preprocessing**: Parse 37 breed labels and species (cat vs. dog) from filenames across 7,389 images, construct stratified train/validation/test splits, and implement input normalization ($224 \times 224 \times 3$, $[-1, 1]$ scaling) with data augmentation.
2. **Weight Initialization Study**: Train MobileNetV2 from scratch to compare Zero, Random Normal ($\sigma=0.05$), Xavier/Glorot, and He initializations, demonstrating symmetry breaking.
3. **Regularization & Overfitting Analysis**: Evaluate No Regularization, $L_2$ Regularization ($\lambda=10^{-3}$), Dropout ($p=0.5$), and Batch Normalization to analyze generalization gaps.
4. **Batch Normalization (BN) Mechanics**: Derive and compute a step-by-step numerical BN example ($x=[2,4,6,8]$) and observe BN effect on training convergence speed.
5. **Optimization Algorithms**: Compare SGD, SGD with Momentum ($\beta=0.9$), RMSProp, and Adam ($\text{lr}=10^{-3}$) on training loss convergence.
6. **CNN Hyperparameter Sweep**: Conduct one-factor-at-a-time sweeps over Learning Rate ($10^{-3}, 10^{-4}$), Batch Size ($16, 32, 64$), and Dropout Rate ($0.0, 0.25, 0.5$).
7. **Transfer Learning & Fine-Tuning**: Feature extraction with frozen ImageNet-pretrained MobileNetV2 base vs. partial unfreezing (fine-tuning upper layers from block 100 onward with a smaller learning rate $10^{-5}$).
8. **5-Fold Cross-Validation & Model Selection**: Perform genuine 5-fold stratified cross-validation across candidate configurations to select the optimal hyperparameter combination (C2: $\text{lr}=10^{-3}$, Dropout $=0.25$, frozen base) using mean accuracy and standard deviation ($\text{mean} - 0.5 \times \text{SD}$).
9. **Final Evaluation & Error Analysis**: Retrain the selected configuration on the full training pool, evaluate once on the held-out test set ($90.93\%$ accuracy), plot the confusion matrix, and inspect misclassified test images.

## Output

Running the notebook populates the `plots/` folder with 15 plots (`plot1_init_loss.png` through `plot15_misclassified.png`) and saves the training history to `results_store.pkl`.

See `EXECUTION_INSTRUCTIONS.md` to run the code and `DATASET_INFO.md` for dataset setup.
