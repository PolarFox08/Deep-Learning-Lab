# Execution Instructions — Deep Learning Lab 6

Follow these steps to set up the environment, configure dataset paths, and run the Jupyter notebook for Deep Learning Lab 6.

---

## 1. Clone the Repository and Navigate to the Lab Directory

```bash
git clone https://github.com/PolarFox08/Deep-Learning-Lab.git
cd Deep-Learning-Lab/"Deep Learning Lab 6"
```

---

## 2. Install Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

---

## 3. Dataset Setup & Path Configuration

The notebook requires access to the **UCI HAR** dataset and the **UCF101** video subset.

1. **UCI HAR Dataset:**
   * Download the raw inertial signals split folders (`train/` and `test/`) from [UCI ML Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) or [Kaggle](https://www.kaggle.com/datasets/uvaiszufar/uci-har).
   * Ensure `Inertial Signals/` folders and `y_train.txt` / `y_test.txt` are intact.

2. **UCF101 Video Subset:**
   * Download the video dataset from [Kaggle](https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition).

3. **Update Notebook Configuration:**
   * Open `dl-lab-6.ipynb` and locate Cell #2 (Configuration cell).
   * Set `HAR_ROOT` to your local path pointing to the root UCI HAR directory containing `train/` and `test/`.
   * Set `VIDEO_ROOT` to your local path pointing to the UCF101 action subfolders (`Basketball`, `Biking`, `WalkingWithDog`, `TennisSwing`).

```python
HAR_ROOT = "path/to/uci-har"
VIDEO_ROOT = "path/to/ucf101/train"
```

---

## 4. Run the Jupyter Notebook

Launch Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

Open and run:
* `dl-lab-6.ipynb` — End-to-End Study of RNN, LSTM and GRU for Sequence Learning and Video Understanding.

---

## 5. Expected Output

Running the notebook will:
* Execute BPTT numerical verification.
* Train Vanilla RNN, LSTM, and GRU models on raw 128-step HAR inertial sensor signals.
* Generate training curves, confusion matrices, model performance comparisons, and sequence length sweeps.
* Extract spatial features using MobileNetV2 and train CNN-LSTM and CNN-GRU models for video action classification.
* Train an Encoder-Decoder LSTM model for synthetic sequence reversal.
* Automatically populate the `plots/` directory with all 13 figure files (`plot1_signals_vs_time.png` through `vid_gru_conf.png`).
