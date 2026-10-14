# Dataset Information — Deep Learning Lab 8 & 9

This laboratory experiment utilizes the **TensorFlow Flowers (`tf_flowers`)** dataset to evaluate the impact of different attention mechanisms (Channel, Spatial, Sequential, Self-Attention, ECA, CBAM) and their strategic insertion positions within an ImageNet-pretrained **ResNet50** convolutional neural network.

---

## 1. Primary Dataset: TensorFlow Flowers (`tf_flowers`)

### Overview
* **Dataset Name:** TensorFlow Flowers (also known as `flower_photos`)
* **Curator / Host:** TensorFlow Team / Google Brain
* **Source / Repository:** Downloaded directly from Google Cloud Storage:
  * Archive URL: `https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz`
  * Automatically downloaded, extracted, and cached in `~/.keras/datasets/flower_photos/` (or loaded directly from `/kaggle/input/` if running in a Kaggle environment).
* **Total Image Count:** $3{,}670$ natural color RGB images.
* **Spatial Resolution:** Variable raw image resolutions, standardized via bilinear interpolation to $224 \times 224 \times 3$ pixels to match the input specification of ResNet50.
* **Target Classes (5 Flower Categories):**
  * `daisy`: 633 images ($17.25\%$)
  * `dandelion`: 898 images ($24.47\%$)
  * `roses`: 641 images ($17.47\%$)
  * `sunflowers`: 699 images ($19.05\%$)
  * `tulips`: 799 images ($21.77\%$)

### Visual Characteristics and Environmental Challenges
The dataset represents real-world, unconstrained natural imagery exhibiting significant intraclass and interclass variation:
1. **Scale Variations:** Images range from extreme macro close-ups of single flower centers and pistils to expansive wide-angle landscape shots containing dozens of flower heads in full fields.
2. **Viewpoint & Perspective:** Aerial, oblique, profile, and ground-level angles.
3. **Illumination Diversity:** Varying sunlight conditions, overcast diffuse lighting, dark shadow occlusions, and high dynamic range backlighting.
4. **Color Polymorphism:** Classes like `tulips` and `roses` exhibit broad petal coloration (red, pink, yellow, white, orange, variegated), while `dandelion` images include both vibrant yellow blossoms and feathery white spherical seed heads.
5. **Background Clutter & Multi-Object Scenes:** Natural complex scenes with green leaves, stems, soil, mulch, fencing, insects (e.g., ladybugs on petals), human hands, and distant buildings.

These environmental characteristics make `tf_flowers` an ideal benchmark for examining whether attention modules truly localize discriminative foreground floral parts (e.g., petal shapes, disc florets) or inadvertently attend to background context.

---

## 2. Experimental Split Protocol

To ensure rigorous, fair, and reproducible benchmarking across all 16 evaluated architectures, a single stratified train-validation-test split was generated using a fixed random seed (`SEED = 42`):

* **Training Subset ($70\%$):** $2{,}569$ images
* **Validation Subset ($15\%$):** $550$ images
* **Test Subset ($15\%$):** $551$ images (Held-out evaluation set)

### Class Distribution Across Experimental Subsets

| Class Name | Training Count | Validation Count | Test Count | Total Count | Class Share (%) |
|---|:---:|:---:|:---:|:---:|:---:|
| **daisy** | 443 | 95 | 95 | 633 | 17.25% |
| **dandelion** | 629 | 134 | 135 | 898 | 24.47% |
| **roses** | 449 | 96 | 96 | 641 | 17.47% |
| **sunflowers** | 489 | 105 | 105 | 699 | 19.05% |
| **tulips** | 559 | 120 | 120 | 799 | 21.77% |
| **Total (All Classes)** | **2,569** | **550** | **551** | **3,670** | **100.0%** |

### Stratification Verification & Persistence
* The split maintains proportional class representation across all three partitions ($24.5\%$ dandelion, $21.8\%$ tulips, $19.0\%$ sunflowers, $17.5\%$ roses, $17.3\%$ daisy).
* The disjoint partition sets were verified programmatically ($\text{train} \cap \text{val} = \emptyset$, $\text{train} \cap \text{test} = \emptyset$, $\text{val} \cap \text{test} = \emptyset$).
* The full image path and label allocation is persisted to disk in `split.csv`, ensuring that every model in Experiments 1 through 5 is trained and tested on the exact same image instances.

---

## 3. Data Preprocessing & Augmentation Pipeline

### 3.1 Preprocessing (All Subsets)
Every raw input image $I_{\text{raw}}$ is processed through a deterministic pipeline prior to model ingestion:
1. **Resizing:** Converted to 3-channel RGB and resized to $224 \times 224$ pixels using high-quality bilinear interpolation:
   $$I_{\text{resized}} = \operatorname{Resize}_{\text{bilinear}}(I_{\text{raw}}, (224, 224))$$
2. **Channel-Wise Mean Centering (ImageNet Standard):**
   Normalized using `tensorflow.keras.applications.resnet50.preprocess_input`:
   $$I_{\text{norm}}(c) = I_{\text{resized}}(c) - \mu_c^{\text{ImageNet}}, \quad c \in \{R, G, B\}$$
   where RGB channels are converted to BGR orientation and zero-centered with respect to ImageNet dataset statistics ($\mu_R = 123.68, \mu_G = 116.78, \mu_B = 103.94$).

### 3.2 Data Augmentation (Training Set Only)
To mitigate overfitting while fine-tuning the 23M+ parameter ResNet50 backbone on 2,569 training samples, identical on-the-fly stochastic transformations are applied to the training set across all 16 models (`random_state = 42`):
* **Random Horizontal Flip:** Reflection across the vertical axis with $p = 0.5$.
* **Random Rotation:** Uniform random rotation with a factor of $0.04$ ($\approx \pm 14.4^\circ$).
* **Random Zoom:** Bounded scaling of $\pm 10\%$ in spatial dimensions.
* **Random Translation:** Horizontal and vertical shifts bounded by $\pm 10\%$.
* **Random Contrast Adjustment:** Linear contrast variations bounded by $\pm 20\%$.

### 3.3 Validation & Test Inference
Validation and test sets receive **zero data augmentation**. Images are fed in their uncorrupted, preprocessed form ($224 \times 224 \times 3$) to guarantee exact, deterministic metric comparisons.

---

## 4. Role in Attention Mechanism & Interpretability Studies

The `tf_flowers` dataset enables three complementary evaluation dimensions:

1. **Quantitative Disentanglement:**
   Because flower classification relies heavily on fine structural attributes (e.g., petal count, petal curvature, disc center color) as well as broad color histograms, models can be tested to observe whether channel re-weighting (SE/ECA), spatial localization, or long-range feature correlations (Self-Attention) yield measurable accuracy or F1 gains.
2. **Internal Attention Map Probing:**
   The spatial maps $M_s(F) \in [0, 1]^{H \times W}$ and channel importance vectors $M_c(F) \in [0, 1]^C$ are extracted directly during the forward pass. This verifies whether the network actively masks background vegetation or saturates across all spatial coordinates.
3. **Post-Hoc Explanations vs. Internal Attention:**
   Grad-CAM activation maps are generated from the final convolutional layer (`conv5_block3_out`, $7 \times 7 \times 2048$) for both the baseline ResNet50 and the attention models. This provides a clear contrast between post-hoc gradient-based saliency and internal forward-pass feature gating.
