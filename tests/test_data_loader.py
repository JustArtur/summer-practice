import os

from src.data_loader import get_datasets, ensure_directory_exists


def test_get_datasets_shapes(tmp_path):
    data_dir = os.path.join(tmp_path, "data")
    ensure_directory_exists(data_dir)
    x_train, y_train, x_test, y_test = get_datasets(
        dataset_directory=str(data_dir),
        train_subset=256,
        test_subset=128,
        random_state=0
    )

    assert x_train.ndim == 2 and x_train.shape[1] == 28 * 28
    assert x_test.ndim == 2 and x_test.shape[1] == 28 * 28
    assert y_train.ndim == 1 and y_test.ndim == 1
    assert x_train.shape[0] == 256
    assert x_test.shape[0] == 128
    assert x_train.min() >= 0.0 and x_train.max() <= 1.0
