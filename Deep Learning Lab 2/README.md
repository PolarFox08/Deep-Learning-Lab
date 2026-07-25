# CS3807 – Deep Learning Laboratory — Experiment 2
## Multi-Layer Perceptron (MLP) for Multi-Class Image Classification (Fashion-MNIST)

This project implements and evaluates a Multi-Layer Perceptron on the
Fashion-MNIST dataset using TensorFlow/Keras, including automated
hyperparameter optimization with RandomizedSearchCV + SciKeras.

## Contents

| File | Description |
|---|---|
| `mlp_fashion_mnist.py` | Main script: loads data, builds/trains/evaluates the baseline MLP, runs hyperparameter search, retrains and evaluates the optimized model, saves all plots and results. |
| `data_loader.py` | Helper module that loads the Fashion-MNIST `.gz` files into NumPy arrays. |
| `Experiment2_MLP_Fashion_MNIST.ipynb` | Same workflow as the `.py` script, organized as a Jupyter notebook with one section per task. |
| `Experiment_2.tex` | Lab report (LaTeX source) with theory, procedure, results, plots with inference, and discussion. |
| `Experiment2_Lab_Record.docx` | Lab report as a Word document. |
| `requirements.txt` | Python package dependencies. |
| `DATASET_INFO.md` | Details about the Fashion-MNIST dataset and how it is obtained. |
| `EXECUTION_INSTRUCTIONS.md` | Step-by-step instructions to set up and run the project. |

## Quick Start

```bash
pip install -r requirements.txt
python mlp_fashion_mnist.py
```

See `EXECUTION_INSTRUCTIONS.md` for full setup details and `DATASET_INFO.md`
for information on how the dataset is loaded.

## What the code does

1. **Task 1 – Dataset Exploration**: loads Fashion-MNIST, prints shapes, plots 10 sample images and the class distribution.
2. **Task 2 – Preprocessing**: flattens images to 784-length vectors, normalizes pixels to [0, 1], one-hot encodes labels.
3. **Task 3 – Model Construction**: builds a baseline MLP — `784 → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)`.
4. **Task 4 – Training**: compiles with Adam + categorical cross-entropy, trains for 20 epochs (batch size 32), plots accuracy/loss curves.
5. **Task 5 – Evaluation**: computes accuracy, precision, recall, F1-score, confusion matrix, and classification report.
6. **Hyperparameter Optimization**: searches hidden layers, neurons, learning rate, batch size, epochs, optimizer, activation, and dropout using RandomizedSearchCV (5-fold CV) via the SciKeras wrapper.
7. **Final Comparison**: retrains the best configuration on the full training set and compares it against the baseline.

## Output

Running the main script produces:
- `plots/` — all 9 required plots (sample images, class distribution, training/validation accuracy & loss, confusion matrix, hyperparameter search results, best model comparison)
- `results.json` — all numeric results (dataset summary, metrics, best hyperparameters, timings)
