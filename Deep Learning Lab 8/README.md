# Deep Learning Lab 8 — Effect of Attention Type and Attention Position in a Pretrained ResNet50 (Experiments 8 & 9)

**Course:** CS3807 -- Deep Learning Laboratory  
**Degree & Branch:** B.Tech Artificial Intelligence & Data Science, Semester V  
**AY:** 2026--27  

---

## Overview

This laboratory experiment systematically investigates how distinct attention mechanisms modify an ImageNet-pretrained **ResNet50** convolutional neural network and how the **position of attention insertion** within the feature hierarchy governs transfer learning performance. Conducted on the **TensorFlow Flowers (`tf_flowers`)** dataset across five natural flower categories (`daisy`, `dandelion`, `roses`, `sunflowers`, `tulips`), the study explores:

1. **Channel Attention (SE-style):** Squeeze-and-Excitation global average pooling followed by a two-layer bottleneck MLP ($r=16$) to model inter-channel dependencies ($F \in \mathbb{R}^{H \times W \times C} \to M_c(F) \in \mathbb{R}^C$).
2. **Spatial Attention:** Channel-wise pooling (Mean and Max across channels) followed by a $7\times7$ (as well as $3\times3$ and $5\times5$) convolution and sigmoid activation to emphasize informative spatial coordinates ($M_s(F) \in \mathbb{R}^{H \times W \times 1}$).
3. **Sequential Channel–Spatial Attention:** Chaining channel and spatial gating in series ($C \to S$ vs. $S \to C$) to determine whether finding "what is important" should precede or succeed "where it is important".
4. **Lightweight Self-Attention:** Flattened spatial query-key-value projections via $1\times1$ convolutions, scaled dot-product attention ($A = \operatorname{softmax}(QK^T/\sqrt{d})$), and a learnable residual scaling parameter $\gamma$ to capture global contextual interactions.
5. **Combined Attention:** A three-stage sequential cascade ($C \to S \to \text{Self}$) incorporating all three modalities.
6. **Additional Established Mechanisms:** Efficient Channel Attention (**ECA**, utilizing local 1D convolutions without dimensionality reduction) and Convolutional Block Attention Module (**CBAM**, combining dual-pooled channel MLPs with spatial convolutions).
7. **Insertion Position Analysis:** Strategically evaluating insertion across four structural stages of ResNet50:
   * $P_1$: After `conv2_x` ($56 \times 56 \times 256$, early low-level edges/textures)
   * $P_2$: After `conv3_x` ($28 \times 28 \times 512$, intermediate local patterns)
   * $P_3$: After `conv4_x` ($14 \times 14 \times 1024$, mid-level structural features)
   * $P_4$: After `conv5_x` ($7 \times 7 \times 2048$, late high-level semantic representations)

The study couples quantitative classification metrics (**Accuracy**, **Macro Precision**, **Macro Recall**, **Macro F1**) and computational complexity (**parameter counts**, **training time**, **inference latency**) with extensive qualitative interpretability tools (**internal channel weights**, **spatial heatmaps**, **self-attention query responses**, and **post-hoc Grad-CAM saliency maps**).

---

## Contents

| File / Folder | Description |
|---|---|
| `dl-lab-8-9.ipynb` | Main Jupyter notebook implementing dataset loading, fixed stratified 70:15:15 splitting, custom Keras attention layers, 16 model architectures, two-stage transfer learning/fine-tuning, evaluation, complexity profiling, internal attention map visualizations, Grad-CAM generation, and comparative analyses. |
| `DL_Lab8_9_Latex_Report.tex` | Comprehensive LaTeX lab report containing complete theoretical background, mathematical formulations, stage-wise tensor dimensions, quantitative tables, figure inferences, confusion matrices, and 8 discussion questions. |
| `DL_Lab_8.pdf` | Compiled PDF report of Experiments 8 & 9. |
| `DATASET_INFO.md` | Detailed breakdown of the TensorFlow Flowers dataset, stratified 70:15:15 split ($2{,}569$ train / $550$ val / $551$ test), preprocessing, and data augmentation pipeline. |
| `EXECUTION_INSTRUCTIONS.md` | Step-by-step setup guide, dependencies, local and Kaggle execution workflows, and generated artifact descriptions. |
| `requirements.txt` | Python package dependencies required to run the notebook. |
| `plots/` | Directory containing all 20 publication-ready generated figures (training/validation curves, mechanism bar charts, position charts, accuracy-complexity trade-offs, confusion matrices, internal attention maps, feature maps, and Grad-CAM overlays). |

---

## Tasks & Experiments Covered

1. **Dataset Pipeline & Fixed Stratified Split:** Extract $3{,}670$ natural flower images, resize to $224 \times 224 \times 3$, apply ImageNet channel centering, and construct a single reproducible $70\% : 15\% : 15\%$ stratified split (`split.csv`, `SEED = 42`).
2. **ResNet50 Feature Hierarchy & Stage Taps:** Define non-destructive, shape-preserving attention insertion interfaces at stages $P_1$, $P_2$, $P_3$, and $P_4$.
3. **Two-Stage Transfer Learning Protocol:** Execute Stage 1 (frozen backbone, 5 epochs, Adam $\text{lr} = 10^{-3}$) followed by Stage 2 (unfrozen `conv5_x`, 10 epochs, Adam $\text{lr} = 10^{-4}$), restoring optimal weights via validation-loss checkpoints.
4. **Main Experiment 1 (Attention-Type Ablation at $P_3$):** Evaluate baseline ResNet50 (A0) against Channel (A1), Spatial (A2), Channel+Spatial (A3), Self-Attention (A4), and Combined (A5) attention.
5. **Main Experiment 2 (Attention-Position Ablation):** Insert identical Channel$\to$Spatial modules across stages $P_1$ (B1), $P_2$ (B2), $P_3$ (B3/A3), and $P_4$ (B4) to trace the impact of feature depth and spatial resolution.
6. **Main Experiment 3 (Multiple Attention Positions):** Benchmark single-stage insertion against multi-stage cascades: $P_3$ only, $P_4$ only, $P_3+P_4$ (C1), and $P_2+P_3+P_4$ (C2).
7. **Main Experiment 4 (Attention Ordering at $P_3$):** Compare Channel-first ($C \to S$, A3) against Spatial-first ($S \to C$, D1) sequential gating under identical parameter budgets.
8. **Main Experiment 5 (Spatial Kernel Size Sweep at $P_3$):** Analyze the receptive field trade-off for spatial attention using $3\times3$ (E1), $5\times5$ (E2), and $7\times7$ (A2) convolution kernels.
9. **Additional Attention Mechanisms (ECA & CBAM at $P_3$):** Compare lightweight Efficient Channel Attention (F1, 5 parameters) and standard CBAM (F2, 132K parameters).
10. **Accuracy–Complexity Trade-Off Profiling:** Contrast classification metrics against total parameters, trainable parameters, training duration, and per-image inference latency.
11. **Confusion Matrix & Per-Class Shift Analysis:** Contrast Baseline (A0) against the top-performing validation attention model (A4: Self-Attention) across all 5 classes.
12. **Internal Attention Map Diagnostics:** Visualize top-20 channel weights, spatial heatmaps/overlays, self-attention query-response matrices, and pre-/post-attention feature activations.
13. **Grad-CAM Saliency & Failure Analysis:** Generate post-hoc Grad-CAM explanations from `conv5_block3_out`, quantify spatial coverage and entropy, and analyze failure modes on misclassified test samples.

---

## Results Summary

### 1. Primary Model Comparison: Attention-Type Ablation at Stage $P_3$

All models evaluated on the held-out test set ($551$ images) using the identical protocol and input split:

| Model ID | Model Architecture | Position | Test Acc. $\uparrow$ | Macro Prec. | Macro Recall | Macro F1 $\uparrow$ | Total Params | Attn. Params | Train (s) | Infer. (ms) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A0** | **Baseline ResNet50 (No Attention)** | -- | **0.9510** | **0.9519** | **0.9503** | **0.9509** | **23,597,957** | **0** | **343.2** | **3.904** |
| **A1** | Channel Attention (SE-style) | $P_3$ | 0.9456 | 0.9455 | 0.9473 | 0.9457 | 23,730,117 | 132,160 | 338.9 | 3.962 |
| **A2** | Spatial Attention ($7\times7$) | $P_3$ | 0.9165 | 0.9200 | 0.9170 | 0.9181 | 23,598,056 | 99 | 337.6 | 3.953 |
| **A3** | Channel + Spatial ($C \to S$) | $P_3$ | 0.9292 | 0.9302 | 0.9278 | 0.9284 | 23,730,216 | 132,259 | 340.4 | 3.962 |
| **A4** | Lightweight Self-Attention | $P_3$ | 0.9437 | 0.9507 | 0.9410 | 0.9444 | 24,908,678 | 1,310,721 | 348.0 | 4.243 |
| **A5** | Combined ($C \to S \to \text{Self}$) | $P_3$ | 0.9201 | 0.9191 | 0.9221 | 0.9199 | 25,040,937 | 1,442,980 | 356.0 | 4.276 |

---

### 2. Attention-Position Ablation (Channel $\to$ Spatial Module)

Evaluating identical Channel $\to$ Spatial modules ($r=16, 7\times7$ kernel) across ResNet50 structural stages:

| Exp. ID | Position | Stage Location | Resolution | Channels | Test Acc. $\uparrow$ | Macro F1 $\uparrow$ | Total Params | Attn. Params | Infer. (ms) |
|:---:|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B1** | $P_1$ | After `conv2_x` | $56 \times 56$ | 256 | 0.9111 | 0.9122 | 23,606,520 | 8,563 | 4.171 |
| **B2** | $P_2$ | After `conv3_x` | $28 \times 28$ | 512 | 0.9419 | 0.9416 | 23,631,368 | 33,411 | 4.046 |
| **B3** (=A3) | $P_3$ | After `conv4_x` | $14 \times 14$ | 1024 | 0.9292 | 0.9284 | 23,730,216 | 132,259 | 3.962 |
| **B4** | $P_4$ | After `conv5_x` | $7 \times 7$ | 2048 | **0.9492** | **0.9499** | 24,124,520 | 526,563 | 3.981 |

---

### 3. Multi-Position, Attention Ordering, Kernel Size & Additional Mechanisms

| Exp. / Model | Variant / Mechanism | Position | Test Acc. $\uparrow$ | Macro F1 $\uparrow$ | Total Params | Attn. Params | Infer. (ms) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **C1** | Multi-Stage ($P_3 + P_4$) | $P_3, P_4$ | 0.9328 | 0.9336 | 24,256,779 | 658,822 | 4.009 |
| **C2** | Multi-Stage ($P_2 + P_3 + P_4$) | $P_2, P_3, P_4$ | 0.9256 | 0.9258 | 24,290,190 | 692,233 | 4.075 |
| **D1** | Attention Order: Spatial $\to$ Channel ($S \to C$) | $P_3$ | 0.9492 | 0.9489 | 23,730,216 | 132,259 | 3.982 |
| **E1** | Spatial Kernel: $3 \times 3$ | $P_3$ | 0.9492 | 0.9500 | 23,597,976 | 19 | 3.951 |
| **E2** | Spatial Kernel: $5 \times 5$ | $P_3$ | 0.9274 | 0.9266 | 23,598,008 | 51 | 3.964 |
| **F1** | Efficient Channel Attention (ECA) | $P_3$ | 0.9220 | 0.9213 | 23,597,962 | **5** | 3.942 |
| **F2** | Convolutional Block Attention (CBAM) | $P_3$ | 0.9183 | 0.9177 | 23,730,216 | 132,259 | 3.990 |

---

### 4. Key Findings & Inferences

* **Baseline Dominance on Pretrained Features:** The unaugmented ImageNet-pretrained ResNet50 baseline achieved the highest overall accuracy ($0.9510$, $524/551$ correct) and the fastest inference ($3.904$ ms/img). None of the 15 attention variants surpassed the baseline. The closest variants ($P_4$ Channel+Spatial, $S \to C$ ordering, and $3\times3$ Spatial attention) all reached $0.9492$ (523/551 correct, a single image difference within binomial noise $\approx 0.9\%$).
* **Attention Mechanism Disparities:** Among single mechanisms at $P_3$, SE-style Channel Attention ($0.9456$) and Self-Attention ($0.9437$) preserved accuracy near baseline levels. In contrast, spatial gating ($0.9165$), sequential $C \to S$ ($0.9292$), and combined attention ($0.9201$) suffered noticeable degradation ($-2.2$ to $-3.5$ percentage points).
* **Insertion Position Dynamics:** Early-stage insertion at $P_1$ performed worst ($0.9111$) and was slowest ($4.171$ ms) due to the large $56 \times 56$ feature resolution. Insertion at late semantic stages ($P_4$, $7 \times 7$) performed best ($0.9492$), confirming that attention is most effective when operating on condensed semantic abstractions rather than raw low-level textures.
* **Multi-Stage Redundancy:** Stacking attention at multiple stages ($P_3+P_4$: $0.9328$; $P_2+P_3+P_4$: $0.9256$) failed to improve over single-stage insertion at $P_4$ ($0.9492$), while incurring up to $+692\text{K}$ parameters and higher inference latency.
* **Ordering Sensitivity:** Gating space before channel ($S \to C$, $0.9492$) outperformed channel-before-space ($C \to S$, $0.9292$) by $+2.00$ percentage points under identical parameter counts ($23.73\text{M}$), because applying channel attention first tended to saturate the subsequent spatial map.
* **Spatial Receptive Field Impact:** Smaller spatial kernels ($3\times3$, $0.9492$) substantially outperformed larger kernels ($7\times7$, $0.9165$). Larger receptive fields suppressed larger swaths of the feature map, eliminating useful context.
* **Internal Attention Behavior:**
  * **Channel Attention:** Channel weights were bimodal (mean $\approx 0.59$, std $\approx 0.43$) but nearly static across images (cross-image correlation $> 0.993$), behaving as a fixed global filter rather than an image-dependent gate.
  * **Spatial Attention:** Internal spatial maps were heavily saturated ($>98\%$ of coordinates $>0.5$, mean $\approx 0.88$--$0.99$), acting primarily to attenuate outermost image borders rather than segmenting flower petals.
  * **Self-Attention:** The $196 \times 196$ attention matrix exhibited extreme peakedness (row entropy $0.010$), with all spatial queries attending to a single dominant focal hot-spot.
* **Grad-CAM Explanations vs. Internal Maps:** Grad-CAM saliency maps were qualitatively similar across all models, concentrating on dominant floral structures. Attention models modified the extent/coverage of the activation (e.g., Self-Attention expanded coverage to $0.2738$ vs. $0.2354$ for baseline) rather than changing the fundamental localization regions.

---

See `EXECUTION_INSTRUCTIONS.md` to run the code and `DATASET_INFO.md` for dataset details.
