# Dataset Information — Fashion-MNIST

## Overview

Fashion-MNIST is a dataset of Zalando article images, intended as a drop-in
replacement for the original MNIST digits dataset. It is widely used as a
benchmark for image classification algorithms.

|Property|Value|
|-|-|
|Total images|70,000|
|Training images|60,000|
|Testing images|10,000|
|Image size|28 × 28 pixels, grayscale|
|Pixel value range|0–255 (raw), scaled to 0–1 during preprocessing|
|Number of classes|10|
|Images per class (training)|6,000 (perfectly balanced)|
|Images per class (testing)|1,000 (perfectly balanced)|

## Class Labels

|Label|Class|
|-|-|
|0|T-shirt/top|
|1|Trouser|
|2|Pullover|
|3|Dress|
|4|Coat|
|5|Sandal|
|6|Shirt|
|7|Sneaker|
|8|Bag|
|9|Ankle boot|

The dataset is automatically downloaded and cached when loading it using `keras.datasets.fashion_mnist` inside the notebook, so no manual download or setup is required.
