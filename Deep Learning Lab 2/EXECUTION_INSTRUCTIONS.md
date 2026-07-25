# Execution Instructions

## 1. Prerequisites

- Python 3.9–3.11
- pip
- (Optional) a virtual environment tool such as `venv` or `conda`

## 2. Set up the environment

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

If `pip install` fails on `scikit-learn` because a newer version is already
present, uninstall it first:

```bash
pip uninstall -y scikit-learn
pip install "scikit-learn<1.6.0"
pip install scikeras
```

## 3. Get the dataset

**Option A — automatic (recommended, used by the notebook):**
No action needed. `keras.datasets.fashion_mnist.load_data()` downloads and
caches the dataset automatically the first time it is called.

**Option B — manual (used by `mlp_fashion_mnist.py` via `data_loader.py`):**
Download the four gzip files listed in `DATASET_INFO.md` from
https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion
and place them in a folder named `data/` in the project directory:

```
project/
├── data/
│   ├── train-images-idx3-ubyte.gz
│   ├── train-labels-idx1-ubyte.gz
│   ├── t10k-images-idx3-ubyte.gz
│   └── t10k-labels-idx1-ubyte.gz
├── data_loader.py
├── mlp_fashion_mnist.py
└── ...
```

## 4. Run the project

**As a script:**

```bash
python mlp_fashion_mnist.py
```

This runs all tasks end-to-end (dataset exploration, preprocessing, model
construction, training, evaluation, hyperparameter search, retraining, and
final comparison) and saves:
- All 9 plots to a `plots/` folder
- All numeric results to `results.json`

**As a notebook:**

```bash
jupyter notebook Experiment2_MLP_Fashion_MNIST.ipynb
```

Run all cells in order (Cell → Run All). Each section is labeled by task
(Task 1 through Task 5, then Hyperparameter Optimization, Retraining, and
Evaluation).

## 5. Expected running time

- Baseline model training (20 epochs, full 60,000-image training set):
  a few minutes on CPU, faster with a GPU.
- Hyperparameter search (RandomizedSearchCV, `n_iter=10`, 5-fold CV):
  the most time-consuming step, since it trains up to 50 models. Running
  time depends heavily on the sampled batch sizes/epochs and available
  hardware; using a GPU is recommended. On CPU-only machines, consider
  reducing `n_iter` or using a subsample of the training data for the
  search phase to keep runtime manageable.


