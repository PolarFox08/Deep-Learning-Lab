# Execution Instructions — Deep Learning Lab 8 & 9

Follow these instructions to configure the environment, download the dataset, and execute the complete experimental pipeline for Deep Learning Lab 8 & 9 (**Effect of Attention Type and Attention Position in a Pretrained ResNet50**).

---

## 1. Clone the Repository and Navigate to the Lab Directory

Open your terminal or PowerShell prompt and navigate into the `Deep Learning Lab 8` folder:

```bash
git clone https://github.com/PolarFox08/Deep-Learning-Lab.git
cd Deep-Learning-Lab/"Deep Learning Lab 8"
```

---

## 2. Environment Setup & Dependency Installation

It is recommended to run the experiments in a clean virtual environment using Python 3.10, 3.11, 3.12, or 3.13.

### 2.1 Create and Activate a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2.2 Install Dependencies

Install all required packages from `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.3 Hardware Verification (GPU Acceleration)

Because this lab trains 16 distinct ResNet50-based architectures with two-stage transfer learning (frozen backbone followed by fine-tuning on `conv5_x`), GPU acceleration (NVIDIA CUDA / TensorRT or Apple Metal) is strongly recommended. Verify your GPU configuration:

```bash
python -c "import tensorflow as tf; print('TensorFlow Version:', tf.__version__); print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

*Note: On a single NVIDIA Tesla T4 or RTX 3060/4060 GPU, each model trains in approximately 5.5 to 6 minutes (total wall-clock time $\approx 90$ minutes for all 16 models). The code also runs on CPU, though training time will be significantly longer.*

---

## 3. Dataset Setup

The experiment utilizes the standard **TensorFlow Flowers (`tf_flowers`)** dataset ($3{,}670$ natural images, 5 classes).

* **Automatic Download:** When executing Cell 2 (`Section 1`), `tf.keras.utils.get_file()` automatically downloads `flower_photos.tgz` from `https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz` and extracts it into `~/.keras/datasets/flower_photos/`.
* **Kaggle Environment Compatibility:** If executing on Kaggle, the notebook automatically discovers pre-attached flower datasets under `/kaggle/input/` before falling back to network download.
* **Stratified Split Persistence:** The notebook creates a single fixed $70\% : 15\% : 15\%$ stratified split (`SEED = 42`) and writes `split.csv` to ensure identical image allocations across all models.

---

## 4. Run the Jupyter Notebook

### Option A: Local Execution
Launch Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

Open:
* **`dl-lab-8-9.ipynb`** — Experiments 8 & 9: Effect of Attention Type and Attention Position in a Pretrained ResNet50.

Select **Kernel $\rightarrow$ Restart & Run All** to run all cells sequentially.

### Option B: Kaggle Execution
1. Upload `dl-lab-8-9.ipynb` to a new Kaggle Notebook.
2. In the right-hand panel under **Settings**:
   * **Accelerator:** `GPU T4 x 2` or `GPU P100`
   * **Internet:** `On` (required once to fetch `flower_photos.tgz` and ImageNet pretrained weights).
3. The notebook writes intermediate checkpoints and generated plots to `/kaggle/working/outputs/`. With `RESUME = True`, if a session disconnects, previously completed models will be automatically loaded from disk without re-training.

---

## 5. Notebook Execution Workflow

The notebook contains 24 modular sections:

1. **Section 0 — Setup and Global Configuration:**
   * Seeds: `SEED = 42` applied to Python, NumPy, and TensorFlow.
   * Image resolution: $224 \times 224 \times 3$.
   * Training hyperparameters: `BATCH_SIZE = 32`, `HEAD_EPOCHS = 5`, `FT_EPOCHS = 10`, `LR_HEAD = 1e-3`, `LR_FT = 1e-4`, `DROPOUT = 0.3`, `REDUCTION = 16`.
2. **Section 1 — Dataset Loading & Fixed Stratified Split:**
   * Reads all $3{,}670$ images once, resizes them, caches them in RAM using a multi-threaded executor (`ThreadPoolExecutor(8)`), and persists the stratified 70:15:15 split to `split.csv`.
   * Sample visualization of 6 representative flower images per class (`plots/v01_sample_images.png`).
3. **Section 2 — Attention Layer Definitions:**
   * Implements custom Keras layers preserving feature shape $(H \times W \times C)$:
     * `ChannelAttention`: Squeeze-and-Excitation MLP with reduction ratio $r=16$.
     * `SpatialAttention`: Channel pooling (Mean + Max) followed by Conv2D ($7\times7$, $5\times5$, or $3\times3$) and Sigmoid.
     * `ChannelSpatialAttention`: Sequential $C \to S$ or $S \to C$ refinement.
     * `SelfAttention`: QKV projections ($1\times1$ Conv2D), scaled dot-product attention, and learnable residual scale parameter $\gamma$.
     * `ECAAttention`: Efficient Channel Attention using 1D convolution ($k=5$).
     * `CBAMAttention`: Dual pooling (Avg + Max) through shared MLP + Spatial Attention.
4. **Section 3 — ResNet50 Backbone & Insertion Taps:**
   * Truncates ImageNet-pretrained ResNet50 at candidate stages:
     * $P_1$: after `conv2_block3_out` ($56 \times 56 \times 256$)
     * $P_2$: after `conv3_block4_out` ($28 \times 28 \times 512$)
     * $P_3$: after `conv4_block6_out` ($14 \times 14 \times 1024$)
     * $P_4$: after `conv5_block3_out` ($7 \times 7 \times 2048$)
   * Appends classification head: GlobalAveragePooling2D $\to$ Dropout(0.3) $\to$ Dense(5, linear logits).
5. **Section 4 — Two-Stage Transfer Learning Protocol:**
   * **Stage 1 (Head Warm-up, 5 Epochs):** Pretrained backbone frozen; train only newly added attention and classifier parameters with Adam ($\alpha = 10^{-3}$).
   * **Stage 2 (Fine-Tuning, 10 Epochs):** Unfreeze `conv5_x` block (BatchNorm layers remain frozen in inference mode); train with reduced learning rate Adam ($\alpha = 10^{-4}$).
   * **Model Selection:** Automatically restores weights from the epoch achieving minimum validation loss.
6. **Sections 5–9 — Five Core Experiments:**
   * **Exp 1 (Attention Type Ablation at $P_3$):** A0 (Baseline), A1 (Channel), A2 (Spatial $7\times7$), A3 (Channel+Spatial), A4 (Self-Attention), A5 (Combined).
   * **Exp 2 (Attention Position Ablation):** B1 ($P_1$), B2 ($P_2$), B3 (=A3, $P_3$), B4 ($P_4$).
   * **Exp 3 (Multiple Attention Positions):** $P_3$ only, $P_4$ only, C1 ($P_3+P_4$), C2 ($P_2+P_3+P_4$).
   * **Exp 4 (Attention Ordering at $P_3$):** A3 ($C \to S$) vs. D1 ($S \to C$).
   * **Exp 5 (Spatial Kernel Size at $P_3$):** E1 ($3\times3$), E2 ($5\times5$), A2 ($7\times7$).
7. **Section 10 — Additional Attention Mechanisms:**
   * F1 (ECA at $P_3$) and F2 (CBAM at $P_3$).
8. **Sections 11–14 — Results Compilation, Graphs & Confusion Matrices:**
   * Master quantitative comparison tables.
   * Plots training and validation curves, parameter trade-offs, and confusion matrices.
9. **Sections 15–22 — Qualitative Analysis & Explainability:**
   * Feature map activations before and after attention.
   * Channel attention weight bar charts and cross-image correlations.
   * Internal spatial attention heatmaps and overlays.
   * Self-attention spatial response maps.
   * Grad-CAM post-hoc visualization and quantitative coverage/entropy profiling.
   * Qualitative failure mode comparison (3 correct vs. 2 incorrect classifications).
10. **Section 23 — Output Export:**
    * Saves all metrics, CSV tables, and publication-ready figures.

---

## 6. Generated Output Artifacts

Execution generates all 20 figures saved in the `plots/` directory:

| Filename | Description |
|---|---|
| `v01_sample_images.png` | Sample $224 \times 224$ training images across all 5 flower classes (6 per class). |
| `v_curves_accuracy_exp1.png` | Training and validation accuracy curves per epoch for Exp 1 models (A0–A5). |
| `v_curves_loss_exp1.png` | Categorical cross-entropy training/validation loss curves for Exp 1 models (A0–A5). |
| `v_curves_val_overlay_exp1.png` | Comparative validation accuracy and loss overlaid across A0–A5. |
| `v_curves_accuracy_exp2.png` | Training and validation accuracy curves for position ablation models (B1–B4). |
| `v_curves_val_overlay_exp2.png` | Comparative validation accuracy and loss overlaid across B1–B4. |
| `v_bar_mech_accuracy.png` | Test accuracy comparison bar chart across all attention mechanisms at $P_3$. |
| `v_bar_mech_f1.png` | Test Macro F1 comparison bar chart across all attention mechanisms at $P_3$. |
| `v_bar_position.png` | Test accuracy and Macro F1 comparison across insertion positions ($P_1, P_2, P_3, P_4$, multi-stage, baseline). |
| `v_scatter_tradeoffs.png` | Accuracy vs. Total Parameters (left) and Macro F1 vs. Inference Time (right). |
| `v_confusion_matrices.png` | Test set confusion matrices: Baseline A0 vs. Selected Self-Attention A4. |
| `v_channel_weights_A1.png` | Top-20 channel attention weights for A1 at $P_3$ across representative test images. |
| `v_channel_weights_A3.png` | Top-20 channel attention weights for A3 ($C \to S$) at $P_3$ across representative test images. |
| `v_spatial_maps_A2.png` | Internal spatial attention heatmaps and image overlays for A2 (Spatial $7\times7$). |
| `v_spatial_maps_A3.png` | Internal spatial attention heatmaps and image overlays for A3 (Channel+Spatial). |
| `v_self_attention.png` | Self-attention spatial response maps for A4 (Mean, Max, Center-query attention). |
| `v_featuremaps_1.png` | Intermediate feature maps before vs. after attention module for A4 at $P_3$. |
| `v_spatial_maps_P2_P3_P4.png` | Stage-wise spatial attention heatmaps ($M_s^{P_2}$, $M_s^{P_3}$, $M_s^{P_4}$) across flower classes. |
| `v_gradcam_layout.png` | Grad-CAM overlays comparing Baseline A0, Channel A1, Channel+Spatial A3, and Self-Attention A4. |
| `v_qualitative_comparison.png` | Side-by-side qualitative analysis of 3 correctly classified and 2 misclassified test images. |
