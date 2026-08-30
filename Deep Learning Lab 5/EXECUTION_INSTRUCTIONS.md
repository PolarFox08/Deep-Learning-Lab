# Execution Instructions

To set up and run the code for Deep Learning Lab 5, follow these steps:

## 1. Clone the Repository and Navigate to the Lab Directory

```bash
git clone https://github.com/PolarFox08/Deep-Learning-Lab.git
cd Deep-Learning-Lab/"Deep Learning Lab 5"
```

## 2. Install Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

## 3. Dataset Setup

The notebook expects the Oxford-IIIT Pet Dataset images to be located at:
* Local / Kaggle path: `/kaggle/input/datasets/tanlikesmath/the-oxfordiiit-pet-dataset/images/images` (or update `DATA_ROOT` in Section 1 of `dl-lab-5.ipynb` to match your local dataset path).

If downloading manually or running outside Kaggle:
1. Download the Oxford-IIIT Pet Dataset from [Kaggle](https://www.kaggle.com/datasets/tanlikesmath/the-oxfordiiit-pet-dataset) or the [Oxford VGG website](https://www.robots.ox.ac.uk/~vgg/data/pets/).
2. Extract the `images` folder.
3. Update the `DATA_ROOT` variable in `dl-lab-5.ipynb` cell #4 to point to your local image directory path.

## 4. Run the Jupyter Notebook

Launch Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

Open and run:
* `dl-lab-5.ipynb` — Comprehensive Study of CNN Training, Regularization, Optimization, Hyperparameter Tuning, Transfer Learning and Cross-Validation.

*Note:* Pretrained MobileNetV2 ImageNet weights are automatically fetched by TensorFlow on the first run, so an active internet connection is required initially. Generated figures will be saved automatically into the `plots/` directory, and training logs will be persisted to `results_store.pkl`.
