# Dataset Information

## CIFAR-10

* **Source:** [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html) (Python pickled version, "CIFAR-10 python")
* **Training images:** 50,000
* **Testing images:** 10,000
* **Classes:** 10 -- `airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`
* **Image size:** $32 \times 32 \times 3$ (RGB)

## Format

The dataset is distributed as a folder `cifar-10-batches-py/` containing:

```
data_batch_1
data_batch_2
data_batch_3
data_batch_4
data_batch_5
test_batch
batches.meta
readme.html
```

Each `data_batch_*` / `test_batch` file is a Python pickle containing 10,000 images (flattened, 3072 bytes each: 1024 red + 1024 green + 1024 blue, channel-major) and their integer labels. `batches.meta` contains the class name list.

## Setup for this Notebook

1. Download and extract the CIFAR-10 python archive from the link above.
2. Place the extracted `cifar-10-batches-py` folder so the notebook can find it at:
   ```
   /content/cifar-10-batches-py
   ```
   (this is the default path used in the notebook -- update the `folder` variable in the "Load CIFAR-10" cell if running outside Colab or storing the data elsewhere).
3. The notebook loads all five training batches and concatenates them, loads the test batch separately, and reshapes each row directly to `(3, 32, 32)` -- no channel transpose is needed since the raw byte layout is already channel-major, which matches the `(N, C, H, W)` format PyTorch convolution layers expect.

## Class Distribution

CIFAR-10 is perfectly balanced: each of the 10 classes has exactly 5,000 training images and 1,000 test images.
