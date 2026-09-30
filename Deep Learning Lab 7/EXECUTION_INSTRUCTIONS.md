# Execution Instructions — Deep Learning Lab 7

Follow these instructions to set up the environment, verify the dataset, and execute the Jupyter notebook for Deep Learning Lab 7.

---

## 1. Clone the Repository and Navigate to the Lab Directory

Open PowerShell or terminal and navigate into the `Deep Learning Lab 7` directory:

```bash
git clone https://github.com/PolarFox08/Deep-Learning-Lab.git
cd Deep-Learning-Lab/"Deep Learning Lab 7"
```

---

## 2. Environment Setup & Dependency Installation

It is recommended to use a virtual environment (Python 3.10 to 3.12). Install all required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### Hardware Verification (Optional GPU Acceleration)
To verify that TensorFlow recognizes your GPU (e.g., NVIDIA CUDA / TensorRT support):

```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

*Note: The notebook runs efficiently on either CPU or GPU. GPU acceleration provides substantial speedups during Convolutional AE and VAE training.*

---

## 3. Dataset Setup

The experiment uses the standard **MNIST Handwritten Digit Dataset**.
* **Automatic Download:** When running the notebook, `tensorflow.keras.datasets.mnist.load_data()` will automatically download the dataset from `https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz` and cache it locally in `~/.keras/datasets/`.
* **No Manual Configuration Required:** Unlike labs requiring external multi-GB directories, Lab 7 does not require manual file path configuration.

---

## 4. Run the Jupyter Notebook

Launch Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

Open and run:
* **`DL Lab 7 CODE.ipynb`** — End-to-End Study of Autoencoders, Convolutional Autoencoders, Denoising Autoencoders and Variational Autoencoders.

Select **Kernel $\rightarrow$ Restart & Run All** to execute all cells sequentially from top to bottom.

---

## 5. Notebook Execution Workflow

The notebook is divided into modular sections:

1. **0. Imports and Configuration:** Sets random seeds (`SEED = 42`), visualization aesthetics (`seaborn whitegrid`), and sets training hyperparameters ($N_{\text{train}}=10,000$, $N_{\text{test}}=2,000$, batch size 128, Adam $\text{lr}=10^{-3}$).
2. **3. Dataset Loading & Stratification:** Loads full MNIST, normalizes pixel intensities to $[0.0, 1.0]$, performs stratified 10,000-train and 2,000-test extraction, and confirms balanced class counts.
3. **6–9. Fully Connected Autoencoder:**
   * Architecture: $784 \to 128 \to 32 \to 16 \to 32 \to 128 \to 784$ (211,040 parameters).
   * Trains for 20 epochs with Binary Cross-Entropy loss.
   * Computes test metrics: MSE ($0.0210$), MAE ($0.0563$), SSIM ($0.7621$).
   * Generates Plot 1 (Original vs. Reconstructed) and Plot 2 (Train/Validation Loss curves).
4. **10–11. Convolutional Autoencoder (CAE):**
   * Preserves spatial dimensions using Conv2D ($3\times3$) and MaxPooling2D in encoder, and UpSampling2D + Conv2D in decoder (74,497 parameters).
   * Evaluates sharp reconstruction metrics: MSE ($0.0026$), MAE ($0.0150$), SSIM ($0.9746$).
   * Generates Plot 3 (FC-AE vs. Conv-AE visual comparison).
5. **12–14. Denoising Convolutional Autoencoder:**
   * Injects additive Gaussian noise $\tilde{x} = \operatorname{clip}(x + \mathcal{N}(0, \sigma^2), 0, 1)$.
   * Trains model on $\sigma = 0.2$ with clean target images.
   * Generates Plot 4 (Clean vs. Noisy vs. Denoised) and Plot 5 (Noise sensitivity across $\sigma \in \{0.1, 0.2, 0.3\}$).
6. **15–20. Variational Autoencoder (VAE):**
   * Encoder outputs mean $\mu$ and log-variance $\log\sigma^2$ in 2D latent space ($z \in \mathbb{R}^2$).
   * Implements Reparameterization Trick ($z = \mu + \sigma \odot \epsilon, \epsilon \sim \mathcal{N}(0, I)$).
   * Minimizes total loss: $\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{rec}} + \mathcal{L}_{\text{KL}}$.
   * Generates Plot 6 (2D latent space scatter plot color-coded by digit), Plot 7 ($5 \times 5$ grid of random generations), and Plot 8 (smooth latent-space trajectory interpolating between digit '0' and '9').
7. **21–25. Model Comparisons & Error Diagnostics:**
   * Cross-model consolidated metrics table.
   * Generates Plot 9 (Reconstruction error distribution histogram).
   * Generates High-Error Analysis Plot (inspects the 5 worst test samples, highlighting loop/stroke intersections in digit 8).
8. **27. Latent Dimension Study:** Sweeps latent dimension $d_z \in \{2, 8, 16, 32\}$ on FC-AE and generates Plot 10.
9. **28. Discussion Questions:** Conceptual answers to 24 foundational theoretical questions.
10. **Additional Exercises (Exercises 1–8):**
    * **Ex 1:** Conv-AE latent dimension sweep ($d_z \in \{4, 8, 16, 32\}$) $\to$ `add_plots/q1.png`.
    * **Ex 2:** Gaussian vs. Salt-and-Pepper noise comparison $\to$ `add_plots/q2.png`.
    * **Ex 3:** Robustness comparison between models trained at $\sigma=0.1$ vs. $\sigma=0.3$.
    * **Ex 4:** UpSampling2D vs. Conv2DTranspose decoder comparison $\to$ `add_plots/q4.png`.
    * **Ex 5:** VAE latent dimension comparison ($d_z = 2$ vs. $d_z = 8$) $\to$ `add_plots/q5a.png`--`q5d.png`.
    * **Ex 6:** $\beta$-VAE analysis on KL weight ($\beta \in \{0.1, 1.0, 10.0, 50.0\}$) and latent collapse $\to$ `add_plots/q6.png`.
    * **Ex 7:** Large $10 \times 10$ and $15 \times 15$ sample diversity grids $\to$ `add_plots/q7a.png`, `q7b.png`.
    * **Ex 8:** Pairwise digit centroid interpolations $\to$ `add_plots/q8a.png`, `q8b.png`.

---

## 6. Generated Output Artifacts

Running the notebook populates the output directories:

### Core Figures (`plots/`):
* `plot1_orig_recon_fc.png`: Original vs. reconstructed digits for FC-AE.
* `plot2_fc_loss.png`: Training and validation BCE loss convergence.
* `plot3_fc_conv.png`: Side-by-side reconstruction comparison (FC-AE vs. Conv-AE).
* `plot4_denoise.png`: Clean, noisy ($\sigma=0.2$), and denoised digit reconstructions.
* `plot5_noise_quality.png`: MSE, MAE, and SSIM curves across noise levels.
* `plot6_latent_space.png`: 2D VAE latent space colored by digit class.
* `plot7_generated.png`: $5 \times 5$ grid of newly synthesized digits sampled from $\mathcal{N}(0, I)$.
* `plot8_interp.png`: 11-step continuous latent manifold interpolation from digit '0' to '9'.
* `plot9_error_hist.png`: Right-skewed distribution of per-sample MSE on test set.
* `plot_higherror.png`: Analysis of the five worst reconstructed test samples.
* `plot10_latentdim.png`: Latent capacity sweep ($d_z \in \{2, 8, 16, 32\}$).

### Additional Exercise Figures (`add_plots/`):
* `q1.png`: Conv-AE latent dimension sweep ($d_z \in \{4, 8, 16, 32\}$).
* `q2.png`: Gaussian vs. Salt-and-Pepper denoising performance.
* `q4.png`: Conv2DTranspose vs. UpSampling2D visual comparison.
* `q5a.png`, `q5b.png`, `q5c.png`, `q5d.png`: VAE $d_z = 2$ vs. $d_z = 8$ latent structures and active dimension analysis.
* `q6.png`: $\beta$-VAE posterior collapse investigation.
* `q7a.png`, `q7b.png`: Large grid sample diversity ($100$ random samples and $15 \times 15$ latent space manifold map).
* `q8a.png`, `q8b.png`: Inter-digit centroid manifold walks.
