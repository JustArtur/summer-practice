import gzip
import io
import os
import struct
import urllib.request
from typing import Tuple, Optional

import numpy as np

MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def ensure_directory_exists(directory_path: str) -> None:
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path, exist_ok=True)


def _download_file(url: str, destination_path: str) -> None:
    if os.path.isfile(destination_path):
        return
    with urllib.request.urlopen(url) as response:
        data = response.read()
    with open(destination_path, "wb") as f:
        f.write(data)


def download_mnist(dataset_directory: str) -> None:
    ensure_directory_exists(dataset_directory)
    for filename in FILES.values():
        url = MNIST_BASE_URL + filename
        dest = os.path.join(dataset_directory, filename)
        _download_file(url, dest)


def _read_idx_images(gz_path: str) -> np.ndarray:
    with gzip.open(gz_path, "rb") as f:
        content = f.read()
    buffer = io.BytesIO(content)
    magic, num_images, num_rows, num_cols = struct.unpack(
        ">IIII", buffer.read(16)
    )
    if magic != 2051:
        raise ValueError(f"Unexpected magic {magic} in images file {gz_path}")
    image_data = np.frombuffer(buffer.read(), dtype=np.uint8)
    images = image_data.reshape(
        num_images, num_rows * num_cols
    ).astype(np.float32) / 255.0
    return images


def _read_idx_labels(gz_path: str) -> np.ndarray:
    with gzip.open(gz_path, "rb") as f:
        content = f.read()
    buffer = io.BytesIO(content)
    magic, num_items = struct.unpack(">II", buffer.read(8))
    if magic != 2049:
        raise ValueError(f"Unexpected magic {magic} in labels file {gz_path}")
    labels = np.frombuffer(buffer.read(), dtype=np.uint8).astype(np.int64)
    return labels


def load_mnist(
    dataset_directory: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST from local gz files, downloading if needed."""
    download_mnist(dataset_directory)
    train_images_path = os.path.join(dataset_directory, FILES["train_images"])
    train_labels_path = os.path.join(dataset_directory, FILES["train_labels"])
    test_images_path = os.path.join(dataset_directory, FILES["test_images"])
    test_labels_path = os.path.join(dataset_directory, FILES["test_labels"])

    x_train = _read_idx_images(train_images_path)
    y_train = _read_idx_labels(train_labels_path)
    x_test = _read_idx_images(test_images_path)
    y_test = _read_idx_labels(test_labels_path)
    return x_train, y_train, x_test, y_test


def get_datasets(
    dataset_directory: str,
    train_subset: Optional[int] = None,
    test_subset: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened, normalized MNIST datasets.

    With optional subsampling.
    """
    x_train, y_train, x_test, y_test = load_mnist(dataset_directory)

    rng = np.random.RandomState(random_state)

    if train_subset is not None and train_subset < len(x_train):
        indices = rng.choice(len(x_train), size=train_subset, replace=False)
        x_train = x_train[indices]
        y_train = y_train[indices]

    if test_subset is not None and test_subset < len(x_test):
        indices = rng.choice(len(x_test), size=test_subset, replace=False)
        x_test = x_test[indices]
        y_test = y_test[indices]

    return x_train, y_train, x_test, y_test
