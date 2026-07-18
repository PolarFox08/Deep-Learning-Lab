# Single Layer Perceptron — Banknote Authentication

Implementation of a Single Layer Perceptron **from scratch** (no ML framework for the model itself) for binary classification, built for CS3807 – Deep Learning Laboratory, Experiment 1.

The perceptron classifies banknotes as **authentic** or **forged** using four numerical features extracted from wavelet-transformed banknote images.

## Contents

| File | Description |
|---|---|
| `DL_Lab_1.ipynb` | Full notebook: EDA, preprocessing, from-scratch Perceptron, training, evaluation, learning-rate comparison, sklearn comparison |
| `data_banknote_authentication.csv` | Local copy of the dataset (see `DATASET.md`) |
| `requirements.txt` | Python dependencies |
| `DATASET.md` | Dataset description and source |
| `EXECUTION.md` | Setup and run instructions |

## What's implemented

- Exploratory Data Analysis: histograms, correlation heatmap, pairwise scatter plot, boxplots
- Feature normalization (Min-Max scaling) and an 80/20 train-test split
- A `Perceptron` class implemented from scratch:
  - Zero weight/bias initialization
  - Step activation function
  - Forward propagation
  - Perceptron learning rule (`w += η(y − ŷ)x`, `b += η(y − ŷ)`)
- Per-epoch logging of misclassified samples, weights, and bias
- Evaluation: accuracy, precision, recall, F1-score, confusion matrix
- Learning rate comparison (η = 0.001, 0.01, 0.1)
- Comparison against scikit-learn's built-in `Perceptron`
- Optional: 2D decision boundary visualization

## Results

| Metric | Value |
|---|---|
| Accuracy | 0.9891 |
| Precision | 1.0000 |
| Recall | 0.9764 |
| F1-score | 0.9880 |

**From-scratch vs scikit-learn Perceptron (test accuracy):**

| Implementation | Accuracy |
|---|---|
| From-scratch | 0.9891 |
| scikit-learn | 0.9782 |

The small difference is expected — scikit-learn's `Perceptron` shuffles training data every epoch and uses early stopping by default, while the from-scratch version iterates in a fixed order for a fixed 50 epochs, so the two land on slightly different decision boundaries.

**Note on learning rate:** all three tested learning rates (0.001, 0.01, 0.1) produce identical misclassification curves. This is expected for this implementation — since weights start at zero and samples are processed in a fixed order, scaling η only scales the magnitude of the learned weights, not the sign of any prediction, so the same samples get misclassified at every epoch regardless of η.

## Quick start

```bash
git clone <your-repo-url>
cd <your-repo>
pip install -r requirements.txt
jupyter notebook DL_Lab_1.ipynb
```

See `EXECUTION.md` for full setup instructions and `DATASET.md` for dataset details.

## References

1. F. Rosenblatt, "The Perceptron," *Psychological Review*, 1958.
2. I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press, 2016.
3. UCI Machine Learning Repository — Banknote Authentication Dataset.
