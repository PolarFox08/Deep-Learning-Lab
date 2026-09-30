# Dataset Information — Deep Learning Lab 7

This lab utilizes the **MNIST Handwritten Digit Dataset** to investigate unsupervised representation learning, image reconstruction, spatial preservation, noise removal, and generative latent space modeling across deterministic and probabilistic autoencoder architectures.

---

## 1. Primary Dataset: MNIST Handwritten Digits

### Overview
* **Dataset Name:** MNIST (Modified National Institute of Standards and Technology) Database of Handwritten Digits
* **Originators:** Yann LeCun, Corinna Cortes, Christopher J. C. Burges
* **Source / Repository:** Loaded directly via `tf.keras.datasets.mnist` ([Keras API Reference](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist) / [Yann LeCun's Website](https://yann.lecun.com/exdb/mnist/))
* **Total Image Count:** 70,000 grayscale digit images (60,000 official training set, 10,000 official test set)
* **Spatial Resolution:** $28 \times 28$ pixels, single grayscale channel ($28 \times 28 \times 1$)
* **Target Classes (10 Digits):**
  * Digit `0`
  * Digit `1`
  * Digit `2`
  * Digit `3`
  * Digit `4`
  * Digit `5`
  * Digit `6`
  * Digit `7`
  * Digit `8`
  * Digit `9`

### Role in Unsupervised Learning
In this experiment, the image itself serves as both the network input and the reconstruction target:
$$\boxed{x \xrightarrow{\text{Encoder}} z \xrightarrow{\text{Decoder}} \hat{x}}$$

Ground-truth digit labels ($y \in \{0, \dots, 9\}$) are **not** provided to the autoencoders during training. Instead, labels are preserved solely for:
1. Stratified sampling to guarantee balanced class distributions across subsets.
2. Color-coded 2D latent-space manifold visualizations ($z \in \mathbb{R}^2$).
3. Qualitative class-specific error analysis and centroid interpolation.

---

## 2. Experimental Subset & Splitting Protocol

To ensure controlled and reproducible evaluation while maintaining computational feasibility, a stratified subset is drawn using a fixed random seed (`SEED = 42`):

* **Training Subset:** 10,000 images sampled with stratified sampling from the official 60,000-image training split.
* **Test Subset (Held-Out):** 2,000 images sampled with stratified sampling from the official 10,000-image test split.

### Class Distribution in Experimental Subsets

| Digit | Training Count (out of 10,000) | Percentage (%) | Test Count (out of 2,000) |
|:---:|:---:|:---:|:---:|
| **0** | 987 | 9.87% | 196 |
| **1** | 1,124 | 11.24% | 227 |
| **2** | 993 | 9.93% | 206 |
| **3** | 1,022 | 10.22% | 202 |
| **4** | 974 | 9.74% | 196 |
| **5** | 904 | 9.04% | 179 |
| **6** | 986 | 9.86% | 191 |
| **7** | 1,044 | 10.44% | 206 |
| **8** | 975 | 9.75% | 195 |
| **9** | 991 | 9.91% | 202 |
| **Total** | **10,000** | **100.0%** | **2,000** |

The stratified sampling ensures that every digit class represents approximately $9\%$--$11\%$ of the subset, mirroring the natural distribution of the full MNIST dataset.

---

## 3. Data Preprocessing & Input Representations

All images undergo consistent preprocessing prior to ingestion into the models:

1. **Pixel Intensity Normalization:**
   Raw unsigned 8-bit integers $[0, 255]$ are converted to single-precision floating-point numbers and normalized to the unit interval $[0.0, 1.0]$:
   $$x_{\text{norm}} = \frac{x_{\text{raw}}}{255.0} \in [0.0, 1.0]$$

2. **Input Tensor Shapes by Architecture:**
   * **Fully Connected Autoencoder (FC-AE):** Images are flattened into 1D feature vectors:
     $$x_{\text{flat}} \in \mathbb{R}^{N \times 784} \quad (28 \times 28 = 784)$$
   * **Convolutional Autoencoder (CAE) & Denoising CAE:** Spatial 2D structure is preserved:
     $$x_{\text{spatial}} \in \mathbb{R}^{N \times 28 \times 28 \times 1}$$
   * **Variational Autoencoder (VAE):** Evaluated both on spatial tensors $\mathbb{R}^{N \times 28 \times 28 \times 1}$ and encoded into a 2D or 8D latent distribution $\mathcal{N}(\mu, \operatorname{diag}(\sigma^2))$.

---

## 4. Noise Models for Denoising Studies

For the Denoising Autoencoder experiments, artificial corruptions are synthesized from clean test and training sets:

### 1. Additive Gaussian Noise
Additive white Gaussian noise is sampled independently for each pixel:
$$\tilde{x} = \operatorname{clip}(x + n, 0.0, 1.0), \quad n \sim \mathcal{N}(0, \sigma^2)$$

* **Evaluated Noise Standard Deviations:** $\sigma \in \{0.1, 0.2, 0.3\}$
* **Training Noise Level:** The primary Denoising CAE is trained on $\sigma = 0.2$ and tested across $\sigma \in \{0.1, 0.2, 0.3\}$ to assess noise generalization.
* **Dual Noise Comparison (Exercise 3):** Models are trained separately on mild noise ($\sigma = 0.1$) vs. severe noise ($\sigma = 0.3$) and cross-tested.

### 2. Salt-and-Pepper Impulse Noise
Impulse noise corrupts a random fraction $p$ of pixels to extreme values (either minimum $0.0$ or maximum $1.0$ with equal probability):
* **Corruption Fractions Evaluated (Exercise 2):** $p \in \{0.05, 0.10, 0.20\}$
* **Comparison:** Evaluates model capacity to reconstruct strokes corrupted by missing or saturated pixels versus continuous additive perturbations.
