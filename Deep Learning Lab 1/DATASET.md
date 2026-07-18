# Dataset Information

## Banknote Authentication Dataset

**Source:** UCI Machine Learning Repository
**Link:** https://archive.ics.uci.edu/dataset/267/banknote+authentication

### Overview

The dataset was created from images of genuine and forged banknote-like specimens, digitized using an industrial camera typically used for print inspection. Each image was 400×400 pixels, grayscale, with a resolution of about 660 dpi. A Wavelet Transform was applied to extract four statistical features from each image.

| Property | Value |
|---|---|
| Instances | 1372 |
| Features | 4 (all numeric, continuous) |
| Classes | 2 (binary classification) |
| Missing values | None |

### Features

| Feature | Description |
|---|---|
| `variance` | Variance of the Wavelet Transformed image |
| `skewness` | Skewness of the Wavelet Transformed image |
| `curtosis` | Kurtosis of the Wavelet Transformed image |
| `entropy` | Entropy of the image |

### Target

| Value | Class |
|---|---|
| `0` | Authentic banknote |
| `1` | Forged banknote |

### File format

`data_banknote_authentication.csv` — no header row, comma-separated:

```
variance,skewness,curtosis,entropy,class
3.6216,8.6661,-2.8073,-0.44699,0
4.5459,8.1674,-2.4586,-1.4621,0
...
```

When loading, column names are assigned manually since the raw file has no header:

```python
column_names = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
df = pd.read_csv('data_banknote_authentication.csv', header=None, names=column_names)
```

### Notes on separability

Exploratory analysis (see notebook, EDA section) shows `variance` is the most discriminative single feature between classes, with boxplots showing almost no overlap between authentic and forged notes. `curtosis` and `entropy` show substantial overlap between classes, meaning the dataset is *close to* but not *perfectly* linearly separable — which is why the from-scratch perceptron's training error plateaus at a low but non-zero value rather than reaching exactly zero.

### Citation

UCI Machine Learning Repository, Banknote Authentication Data Set. Donated by Volker Lohweg (University of Applied Sciences, Ostwestfalen-Lippe).
