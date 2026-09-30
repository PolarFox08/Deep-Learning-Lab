# Deep Learning Lab 7 — End-to-End Study of Autoencoders, Convolutional Autoencoders, Denoising Autoencoders and Variational Autoencoders

**Course:** CS3807 -- Deep Learning Laboratory  
**Degree & Branch:** B.Tech Artificial Intelligence & Data Science, Semester V  
**AY:** 2026--27  

---

## Overview

This experiment systematically explores unsupervised representation learning, image reconstruction, noise removal, and generative latent space modeling by implementing and evaluating four foundational autoencoder paradigms on the **MNIST** handwritten digit dataset:

1. **Fully Connected Autoencoder (FC-AE):** Dimensionality reduction and bottleneck compression on flattened pixel vectors ($x \in \mathbb{R}^{784} \to z \in \mathbb{R}^{16} \to \hat{x} \in \mathbb{R}^{784}$).
2. **Convolutional Autoencoder (CAE):** Exploiting spatial locality and translation equivariance using 2D convolutions, max pooling, and upsampling to preserve 2D feature geometry ($x \in \mathbb{R}^{28 \times 28 \times 1}$).
3. **Denoising Convolutional Autoencoder (Denoising CAE):** Learning manifold projection and signal recovery by mapping corrupted inputs $\tilde{x} = \operatorname{clip}(x + n, 0, 1)$ to pristine clean targets $x$.
4. **Variational Autoencoder (VAE):** Continuous generative modeling using a probabilistic latent space $q_\phi(z|x) = \mathcal{N}(\mu, \operatorname{diag}(\sigma^2))$ trained via the Reparameterization Trick and Evidence Lower Bound (ELBO) objective ($z \in \mathbb{R}^2$ and $z \in \mathbb{R}^8$).

The experiment evaluates reconstruction quality using pixel-level metrics (**MSE**, **MAE**) and perceptual structural fidelity (**SSIM**), examines latent manifold continuity via smooth interpolation, analyzes high-reconstruction-error failure modes, and investigates $\beta$-VAE posterior collapse.

---

## Contents

| File / Folder | Description |
|---|---|
| `DL Lab 7 CODE.ipynb` | Main Jupyter notebook implementing dataset loading, stratified splitting, model architectures (FC-AE, CAE, Denoising CAE, VAE), training, metric evaluations, latent visualizations, failure case analysis, and all 8 additional exercises. |
| `Experiment_7.tex` | LaTeX lab report source containing complete theoretical background, mathematical derivations (ELBO, KL divergence, Reparameterization Trick), architecture diagrams, quantitative tables, figure inferences, and 24 discussion questions. |
| `DL_Lab_7.pdf` | Compiled PDF report of Experiment 7. |
| `DATASET_INFO.md` | Detailed breakdown of the MNIST dataset, class-balanced stratified subsets (10,000 train / 2,000 test), input tensor representations, and noise models. |
| `EXECUTION_INSTRUCTIONS.md` | Step-by-step setup guide, environment dependencies, execution workflow, and artifact descriptions. |
| `requirements.txt` | Python package dependencies required to run the notebook. |
| `plots/` | Directory containing all 11 core generated figures (reconstructions, training loss curves, noise sweeps, 2D latent space, random generations, manifold interpolation, and error histograms). |
| `add_plots/` | Directory containing all 12 figures generated for the 8 additional exercises (latent dimension sweeps, noise comparisons, decoder variants, VAE capacity, $\beta$-VAE collapse, sample grids, and centroid walks). |

---

## Tasks Covered

1. **Dataset Loading & Stratified Subsetting:** Load MNIST ($70{,}000$ images), normalize pixel intensities to $[0.0, 1.0]$, and construct class-balanced stratified subsets of 10,000 training and 2,000 test images (`SEED = 42`).
2. **Fully Connected Autoencoder:** Construct a symmetric 7-layer bottleneck network ($784 \to 128 \to 32 \to 16 \to 32 \to 128 \to 784$, 211,040 parameters) trained with Binary Cross-Entropy (BCE) loss over 20 epochs.
3. **Reconstruction Metric Evaluation:** Evaluate test set performance using Mean Squared Error (MSE), Mean Absolute Error (MAE), and Structural Similarity Index Measure (SSIM via `skimage.metrics.structural_similarity`).
4. **Convolutional Autoencoder (CAE):** Design a 2D spatial autoencoder with Conv2D and MaxPooling2D layers in the encoder and UpSampling2D + Conv2D in the decoder (74,497 parameters), comparing reconstruction sharpness and parameter efficiency against FC-AE.
5. **Denoising Convolutional Autoencoder:** Corrupt images with additive Gaussian noise ($n \sim \mathcal{N}(0, \sigma^2)$ at $\sigma \in \{0.1, 0.2, 0.3\}$) and train the CAE at $\sigma=0.2$ to map noisy inputs back to clean ground-truth targets.
6. **Noise Robustness Sweep:** Quantify denoising performance across increasing noise levels ($\sigma = 0.1 \to 0.3$), tracking linear degradation in MSE/MAE and graceful preservation of SSIM.
7. **Variational Autoencoder (VAE) Architecture:** Implement a probabilistic encoder outputting mean $\mu \in \mathbb{R}^2$ and log-variance $\log\sigma^2 \in \mathbb{R}^2$, apply the Reparameterization Trick ($z = \mu + \sigma \odot \epsilon, \epsilon \sim \mathcal{N}(0, I)$), and train with a combined reconstruction loss and analytical KL-divergence regularization.
8. **Latent Space Visualization & Manifold Analysis:** Plot encoded test digit embeddings in the 2D latent space color-coded by ground truth, identifying well-separated digit clusters ('0', '1') and overlapping multi-class regions ('3', '8' and '4', '7', '9').
9. **Generative Sampling & Latent Space Interpolation:** Synthesize novel digits by passing random standard normal vectors through the decoder, and perform an 11-step linear manifold walk between digit centroids ($z(\alpha) = (1-\alpha)z_A + \alpha z_B$), observing continuous semantic deformation without broken intermediate frames.
10. **Test Error Distribution & Failure Mode Analysis:** Plot the per-sample reconstruction MSE histogram to observe right-skewness, and isolate the top 5 highest-error test samples (revealing structural difficulty with looped, self-intersecting digits like '8' and closed '0').
11. **Additional Exercises (Exercises 1–8):**
    * *Ex 1:* Conv-AE latent bottleneck sweep ($d_z \in \{4, 8, 16, 32\}$).
    * *Ex 2:* Comparative evaluation of Gaussian vs. Salt-and-Pepper impulse noise.
    * *Ex 3:* Cross-noise level generalization (training at $\sigma=0.1$ vs. $\sigma=0.3$).
    * *Ex 4:* Transposed Convolution (`Conv2DTranspose`) vs. Nearest-Neighbor UpSampling (`UpSampling2D + Conv2D`) decoder comparison.
    * *Ex 5:* VAE latent capacity comparison ($d_z = 2$ vs. $d_z = 8$) and active latent dimension analysis via KL divergence.
    * *Ex 6:* $\beta$-VAE study on the regularization trade-off ($\beta \in \{0.1, 1.0, 10.0, 50.0\}$) and empirical demonstration of posterior collapse at $\beta=50$.
    * *Ex 7:* High-density $10 \times 10$ random generation grid and $15 \times 15$ latent coordinate manifold scanning.
    * *Ex 8:* Inter-cluster centroid latent walks between distinct digit pairs.

---

## Results Summary

### 1. Primary Model Comparison

| Model | Architecture Type | Latent Representation | Parameters | Test MSE $\downarrow$ | Test MAE $\downarrow$ | Mean SSIM $\uparrow$ |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Fully Connected AE** | Dense ($784 \to 16 \to 784$) | 16-D Bottleneck | 211,040 | 0.021037 | 0.056284 | 0.7621 |
| **Convolutional AE** | Conv2D + UpSampling2D | $7 \times 7 \times 64$ Feature Map | **74,497** | **0.002561** | **0.015044** | **0.9746** |
| **Denoising CAE** ($\sigma=0.2$) | Conv2D + UpSampling2D | $7 \times 7 \times 64$ Feature Map | 74,497 | 0.004465 | 0.020965 | 0.9475 |
| **Variational Autoencoder** | Conv2D + Reparameterization | 2-D Gaussian $(\mu, \sigma^2)$ | 70,594 | 0.045464 | 0.108695 | 0.4865 |

*Note: The VAE optimizes the Evidence Lower Bound (ELBO), trading off raw pixel fidelity against latent Gaussian regularization ($\mathcal{L}_{\text{KL}}$). Its 2D bottleneck is also substantially more constrained than the deterministic models.*

### 2. Denoising Robustness Across Noise Levels (Trained at $\sigma=0.2$)

| Input Noise Level ($\sigma$) | Noisy Input MSE | Denoised Output MSE | Denoised Output MAE | Denoised Output SSIM |
|:---:|:---:|:---:|:---:|:---:|
| **0.1** | 0.0098 | 0.003471 | 0.017457 | 0.9608 |
| **0.2** | 0.0385 | 0.004465 | 0.020965 | 0.9475 |
| **0.3** | 0.0821 | 0.006812 | 0.028429 | 0.8945 |

### 3. Additional Key Findings

* **Spatial Locality vs. Flattening:** The Convolutional AE achieves an **$8.2\times$ reduction in MSE** ($0.0026$ vs. $0.0210$) and elevates SSIM from $0.7621$ to $0.9746$, while using **$65\%$ fewer parameters** than the FC-AE. Flattening destroys 2D spatial pixel neighborhoods, forcing dense layers to memorize global correlations.
* **Transposed Convolutions:** Replacing `UpSampling2D + Conv2D` with `Conv2DTranspose` yields identical parameter counts (74,497), marginally superior reconstruction (SSIM $0.9770$ vs. $0.9745$), and slightly faster per-epoch wall-clock speed ($17.1$ s vs. $19.0$ s).
* **Latent Capacity Scaling:** Expanding the VAE latent space from $d_z=2$ to $d_z=8$ increases SSIM from $0.5064$ to $0.7288$ and drops MSE by nearly half ($0.0439 \to 0.0246$). Per-dimension KL analysis reveals that 7 out of 8 dimensions are actively utilized ($KL > 1.3$), while 1 dimension remains uninformative ($KL \approx 0.046$).
* **$\beta$-VAE Posterior Collapse:** Increasing $\beta$ to $50.0$ forces $D_{\text{KL}} \to 0.0001$, dropping active dimensions to 0 and reducing SSIM to $0.1788$, creating blurry average-digit outputs.

---

See `EXECUTION_INSTRUCTIONS.md` to run the code and `DATASET_INFO.md` for dataset details.
