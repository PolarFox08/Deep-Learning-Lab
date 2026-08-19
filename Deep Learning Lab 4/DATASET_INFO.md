# Dataset Information

## CIFAR-10

* **Source:** [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html) (Python pickled version, "CIFAR-10 python")
* **Training images:** 50,000
* **Testing images:** 10,000
* **Classes:** 10 -- `Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck`
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
   (this is the default path used in the notebook -- update the `folder` variable in the data-loading cell if running outside Colab or storing the data elsewhere).
3. The notebook loads all five training batches and concatenates them, loads the test batch separately, reshapes each row to `(3, 32, 32)`, and then **transposes to `(32, 32, 3)`** (channel-last, `NHWC`) since TensorFlow/Keras and the pretrained VGG16 model both expect channel-last input. Pixel values are then normalized to `[0, 1]` and labels one-hot encoded with `to_categorical`.

## Class Distribution

CIFAR-10 is perfectly balanced: each of the 10 classes has exactly 5,000 training images and 1,000 test images.

## Pretrained Model Weights

The notebook also downloads **VGG16 ImageNet weights** (no top/classifier) automatically via `tensorflow.keras.applications.VGG16(weights='imagenet', include_top=False, ...)` on first run -- this requires an internet connection the first time the notebook is executed.
